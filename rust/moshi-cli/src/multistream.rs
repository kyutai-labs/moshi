// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

pub mod client {
    use anyhow::Result;
    use futures_util::{
        stream::{SplitSink, StreamExt},
        SinkExt,
    };
    use std::io::Write;
    use tokio::io::AsyncWriteExt;
    use tokio_tungstenite::tungstenite::protocol::Message;

    type WebSocket = tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >;

    const OPUS_ENCODER_FRAME_SIZE: usize = 960;

    pub struct MsgSender {
        pw: ogg::PacketWriter<'static, Vec<u8>>,
        encoder: opus::Encoder,
        out_pcm: std::collections::VecDeque<f32>,
        out_pcm_buf: Vec<u8>,
        total_data: usize,
        sender: SplitSink<WebSocket, Message>,
    }

    pub(crate) fn write_opus_header<W: std::io::Write>(w: &mut W) -> std::io::Result<()> {
        use byteorder::WriteBytesExt;

        // https://wiki.xiph.org/OggOpus#ID_Header
        w.write_all(b"OpusHead")?;
        w.write_u8(1)?; // version
        w.write_u8(1)?; // channel count
        w.write_u16::<byteorder::LittleEndian>(3840)?; // pre-skip
        w.write_u32::<byteorder::LittleEndian>(48000)?; //  sample-rate in Hz
        w.write_i16::<byteorder::LittleEndian>(0)?; // output gain Q7.8 in dB
        w.write_u8(0)?; // channel map
        Ok(())
    }

    pub(crate) fn write_opus_tags<W: std::io::Write>(w: &mut W) -> std::io::Result<()> {
        use byteorder::WriteBytesExt;

        // https://wiki.xiph.org/OggOpus#Comment_Header
        let vendor = "KyutaiMoshi";
        w.write_all(b"OpusTags")?;
        w.write_u32::<byteorder::LittleEndian>(vendor.len() as u32)?; // vendor string length
        w.write_all(vendor.as_bytes())?; // vendor string, UTF8 encoded
        w.write_u32::<byteorder::LittleEndian>(0u32)?; // number of tags
        Ok(())
    }

    impl MsgSender {
        pub fn new(sender: SplitSink<WebSocket, Message>) -> Result<Self> {
            let encoder = opus::Encoder::new(24000, opus::Channels::Mono, opus::Application::Voip)?;
            // Not sure what the appropriate buffer size would be here.
            let out_pcm_buf = vec![0u8; 50_000];
            let out_pcm = std::collections::VecDeque::with_capacity(2 * OPUS_ENCODER_FRAME_SIZE);

            let all_data = Vec::new();
            let mut pw = ogg::PacketWriter::new(all_data);
            let mut head = Vec::new();
            write_opus_header(&mut head)?;
            pw.write_packet(head, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
            let mut tags = Vec::new();
            write_opus_tags(&mut tags)?;
            pw.write_packet(tags, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
            Ok(Self { pw, encoder, out_pcm, out_pcm_buf, total_data: 0, sender })
        }

        pub async fn send_control(&mut self, control: u8) -> Result<()> {
            let msg = Message::Binary(vec![3u8, control]);
            self.sender.send(msg).await?;
            Ok(())
        }

        pub async fn send_pcm(&mut self, pcm: &[f32]) -> Result<()> {
            self.out_pcm.extend(pcm.iter());
            self.total_data += pcm.len();
            let nchunks = self.out_pcm.len() / OPUS_ENCODER_FRAME_SIZE;
            for _chunk_id in 0..nchunks {
                let mut chunk = Vec::with_capacity(OPUS_ENCODER_FRAME_SIZE);
                for _i in 0..OPUS_ENCODER_FRAME_SIZE {
                    let v = match self.out_pcm.pop_front() {
                        None => anyhow::bail!("unexpected err popping from pcms"),
                        Some(v) => v,
                    };
                    chunk.push(v)
                }
                let size = self.encoder.encode_float(&chunk, &mut self.out_pcm_buf)?;
                if size > 0 {
                    let msg = self.out_pcm_buf[..size].to_vec();
                    self.pw.write_packet(
                        msg,
                        42,
                        ogg::PacketWriteEndInfo::EndPage,
                        self.total_data as u64,
                    )?
                }
                let data = self.pw.inner_mut();
                if !data.is_empty() {
                    let msg: Vec<u8> = [&[1u8], data.as_slice()].concat();
                    let msg = Message::Binary(msg);
                    self.sender.send(msg).await?;
                    data.clear();
                }
            }
            Ok(())
        }
    }

    pub async fn run(host: String, port: usize) -> Result<()> {
        let uri = format!("wss://{host}:{port}/api/chat");
        tracing::info!("connecting to {uri}");
        let (_stream, ad) = crate::audio_io::setup_output_stream(true)?;
        let (_in_stream, input_audio) = crate::audio_io::setup_input_stream()?;
        let connector =
            native_tls::TlsConnector::builder().danger_accept_invalid_certs(true).build()?;
        let (stream, response) = tokio_tungstenite::connect_async_tls_with_config(
            uri,
            None,
            false,
            Some(tokio_tungstenite::Connector::NativeTls(connector)),
        )
        .await?;
        tracing::info!("connected, got {response:?}");
        let (sender, mut receiver) = stream.split();
        let mut sender = MsgSender::new(sender)?;
        let (mut tx, rx) = tokio::io::duplex(100_000);
        tokio::spawn(async move {
            let mut decoder = opus::Decoder::new(24000, opus::Channels::Mono)?;
            let mut pr = ogg::reading::async_api::PacketReader::new(rx);
            let mut pcm_buf = vec![0f32; 24_000 * 120];
            let mut all_pcms = vec![];
            let mut total_size = 0;
            tracing::info!("waiting for audio data");
            while let Some(packet) = pr.next().await {
                let packet = packet?;
                if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags") {
                    continue;
                }
                let size = decoder.decode_float(
                    &packet.data,
                    &mut pcm_buf,
                    /* Forward Error Correction */ false,
                )?;
                if size > 0 {
                    tracing::info!(total_size, size, "received audio");
                    let pcm = &pcm_buf[..size];
                    total_size += size;
                    all_pcms.push(pcm.to_vec());
                    let mut ad = ad.lock().unwrap();
                    ad.push_samples(pcm)?;
                }
            }
            let all_pcms = all_pcms.concat();
            tracing::info!(len = all_pcms.len(), "saving pcms with shape");
            let mut w = std::fs::File::create("received.wav")?;
            moshi::wav::write_pcm_as_wav(&mut w, &all_pcms, 24000)?;
            Ok::<(), anyhow::Error>(())
        });
        tokio::spawn(async move {
            loop {
                let input = input_audio.lock().unwrap().take_all();
                if sender.send_pcm(&input).await.is_err() {
                    break;
                };
                tokio::time::sleep(std::time::Duration::from_millis(20)).await
            }
        });
        while let Some(received) = receiver.next().await {
            match received? {
                Message::Close(_) => break,
                Message::Text(text) => {
                    tracing::error!("unexpected text message {text}");
                    continue;
                }
                Message::Frame(_) | Message::Ping(_) | Message::Pong(_) => continue,
                Message::Binary(bin) => {
                    if bin.is_empty() {
                        continue;
                    }
                    match bin[0] {
                        // Handshake
                        0 => {}
                        // Audio
                        1 => {
                            tx.write_all(&bin[1..]).await?;
                        }
                        2 => {
                            let txt = String::from_utf8_lossy(&bin[1..]);
                            print!("{txt}");
                            std::io::stdout().flush()?;
                        }
                        3 => {
                            tracing::error!("unsupported control message")
                        }
                        4 => {
                            tracing::error!("unsupported metadata message")
                        }
                        mt => {
                            tracing::error!("unexpected message type {mt}");
                            continue;
                        }
                    }
                }
            };
        }
        println!("\n");
        Ok(())
    }
}

pub mod client_tui {
    use super::client::MsgSender;
    use anyhow::Result;
    use futures_util::stream::StreamExt;
    use ratatui::{prelude::*, widgets::*};
    use std::sync::{Arc, Mutex};
    use tokio::io::AsyncWriteExt;
    use tokio::sync::mpsc;
    use tokio_tungstenite::tungstenite::protocol::Message;

    fn initialize_panic_handler() {
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            shutdown().unwrap();
            original_hook(panic_info);
        }));
    }

    fn startup() -> Result<()> {
        crossterm::terminal::enable_raw_mode()?;
        crossterm::execute!(std::io::stderr(), crossterm::terminal::EnterAlternateScreen)?;
        Ok(())
    }

    fn shutdown() -> Result<()> {
        crossterm::execute!(std::io::stderr(), crossterm::terminal::LeaveAlternateScreen)?;
        crossterm::terminal::disable_raw_mode()?;
        Ok(())
    }

    struct Stats {
        recv_messages: usize,
        recv_text_messages: usize,
        recv_audio_messages: usize,
        sent_audio_messages: usize,
    }

    impl Stats {
        fn new() -> Self {
            Self {
                recv_messages: 0,
                recv_text_messages: 0,
                recv_audio_messages: 0,
                sent_audio_messages: 0,
            }
        }
    }

    struct App {
        action_tx: mpsc::UnboundedSender<Action>,
        should_quit: bool,
        ticker: i64,
        state: Arc<Mutex<State>>,
        stats: Arc<Mutex<Stats>>,
        tui_log_state: tui_logger::TuiWidgetState,
        input_audio: Arc<Mutex<crate::audio_io::AudioOutputData_>>,
        output_audio: Arc<Mutex<crate::audio_io::AudioOutputData_>>,
        subs: Arc<Mutex<Vec<String>>>,
        current_db10: u64,
        sender: Arc<tokio::sync::Mutex<MsgSender>>,
    }

    impl App {
        fn current_db10(&mut self) -> u64 {
            let db10 = self.input_audio.lock().unwrap().db10();
            if self.current_db10 as f32 + 1.3 < db10 || db10 < self.current_db10 as f32 - 0.3 {
                self.current_db10 = db10 as u64
            }
            self.current_db10
        }
    }

    fn ui(f: &mut Frame, app: &mut App) {
        let area = f.size();
        let instructions = block::Title::from(Line::from(vec![
            " Quit ".into(),
            "<Q> ".yellow().bold(),
            " Restart ".into(),
            "<R> ".yellow().bold(),
        ]));
        let state = *app.state.lock().unwrap();
        let block = Block::default()
            .title("MoshiMoshi")
            .title_alignment(Alignment::Center)
            .title(instructions.alignment(Alignment::Center).position(block::Position::Bottom))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(8), Constraint::Min(0)])
            .split(block.inner(area));

        let (stats1, stats2) = {
            let input_audio = app.input_audio.lock().unwrap();
            let output_audio = app.output_audio.lock().unwrap();
            let stats = app.stats.lock().unwrap();
            let stats1 = format!(
                "msgs: {}\naudio msgs: {}\ntext msgs: {}\nplay len: {} ({:.1}s)\nplay buf: {} ({:.1}s)\n",
                stats.recv_messages, stats.recv_audio_messages, stats.recv_text_messages,
                output_audio.total_samples(),
                output_audio.total_samples() as f32 / 24000.,
                output_audio.samples_in_buffer(),
                output_audio.samples_in_buffer() as f32 / 24000.,
            );
            let stats2 = format!(
                "audio msgs: {}\nrecd len: {} ({:.1}s)\nrecd buf: {} ({:.1}s)",
                stats.sent_audio_messages,
                input_audio.total_samples(),
                input_audio.total_samples() as f32 / 24000.,
                input_audio.samples_in_buffer(),
                input_audio.samples_in_buffer() as f32 / 24000.,
            );
            (stats1, stats2)
        };
        let header_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(30), Constraint::Length(30)])
            .split(chunks[0]);
        let (header_bg_color, header_fg_color) = match state {
            State::Running => {
                if app.ticker / 4 % 2 == 0 {
                    (Color::Black, Color::Red)
                } else {
                    (Color::Black, Color::Green)
                }
            }
            State::Quit => (Color::Red, Color::White),
        };
        let state = match state {
            State::Running => "\nRUNNING...",
            State::Quit => "\nEXITING...",
        };
        let header_in_block = Block::default()
            .title("state")
            .title_alignment(Alignment::Center)
            .border_style(Style::default().bg(Color::Black).fg(Color::White))
            .borders(Borders::ALL);
        let header_sub_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(header_in_block.inner(header_chunks[0]));

        let header = Paragraph::new(state.bold())
            .style(Style::default().bg(header_bg_color).fg(header_fg_color))
            .alignment(Alignment::Center);
        let stats1 = Paragraph::new(stats1)
            .block(
                Block::default()
                    .title("received")
                    .title_alignment(Alignment::Center)
                    .borders(Borders::ALL),
            )
            .style(Style::default().bg(Color::Black).fg(Color::White))
            .alignment(Alignment::Left);
        let stats2 = Paragraph::new(stats2)
            .block(
                Block::default()
                    .title("sent")
                    .title_alignment(Alignment::Center)
                    .borders(Borders::ALL),
            )
            .style(Style::default().bg(Color::Black).fg(Color::White))
            .alignment(Alignment::Left);
        let bar = BarChart::default()
            .bar_width(1)
            .direction(Direction::Horizontal)
            .bar_style(Style::new().red().on_black())
            .value_style(Style::new().white().bold())
            .label_style(Style::new().white().bold())
            .data(&[("mic", app.current_db10())])
            .max(10);
        f.render_widget(block, area);
        f.render_widget(header_in_block, header_chunks[0]);
        f.render_widget(header, header_sub_chunks[0]);
        f.render_widget(bar, header_sub_chunks[1]);
        f.render_widget(stats1, header_chunks[1]);
        f.render_widget(stats2, header_chunks[2]);
        let chunks = Layout::default()
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(chunks[1]);
        let subs: String = {
            let subs = app.subs.lock().unwrap();
            subs.join("")
        };
        let subs = Paragraph::new(subs)
            .style(Style::default().bg(Color::Black).fg(Color::White))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true });
        f.render_widget(subs, chunks[0]);
        let footer = tui_logger::TuiLoggerSmartWidget::default()
            .style_error(Style::default().fg(Color::Red))
            .style_debug(Style::default().fg(Color::Green))
            .style_warn(Style::default().fg(Color::Yellow))
            .style_trace(Style::default().fg(Color::Magenta))
            .style_info(Style::default().fg(Color::Cyan))
            .output_separator(':')
            .output_timestamp(Some("%H:%M:%S".to_string()))
            .output_level(Some(tui_logger::TuiLoggerLevelOutput::Abbreviated))
            .output_target(true)
            .output_file(true)
            .output_line(true)
            .state(&app.tui_log_state);
        f.render_widget(footer, chunks[1]);
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum State {
        Running,
        Quit,
    }

    #[derive(PartialEq)]
    enum Action {
        None,
        Enter,
        Space,
        Quit,
    }

    async fn update(app: &mut App, msg: Action) -> Result<()> {
        match msg {
            Action::Quit => {
                log::info!("exiting");
                *app.state.lock().unwrap() = State::Quit;
                app.should_quit = true
            }
            Action::None => {}
            Action::Enter => app.sender.lock().await.send_control(0).await?,
            Action::Space => app.sender.lock().await.send_control(1).await?,
        };
        Ok(())
    }

    fn handle_event(tx: mpsc::UnboundedSender<Action>) -> tokio::task::JoinHandle<()> {
        let tick_rate = std::time::Duration::from_millis(250);
        tokio::spawn(async move {
            loop {
                let action = if crossterm::event::poll(tick_rate).unwrap() {
                    if let crossterm::event::Event::Key(key) = crossterm::event::read().unwrap() {
                        if key.kind == crossterm::event::KeyEventKind::Press {
                            if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                                match key.code {
                                    crossterm::event::KeyCode::Char('c' | 'C') => Action::Quit,
                                    _ => Action::None,
                                }
                            } else {
                                match key.code {
                                    crossterm::event::KeyCode::Char('q' | 'Q') => Action::Quit,
                                    crossterm::event::KeyCode::Enter => Action::Enter,
                                    crossterm::event::KeyCode::Char(' ') => Action::Space,
                                    _ => Action::None,
                                }
                            }
                        } else {
                            Action::None
                        }
                    } else {
                        Action::None
                    }
                } else {
                    Action::None
                };
                if tx.send(action).is_err() {
                    break;
                }
            }
        })
    }

    pub async fn run(host: String, port: usize) -> Result<()> {
        let uri = format!("wss://{host}:{port}/api/chat");
        tracing::info!("connecting to {uri}");
        let subs = Arc::new(Mutex::new(vec![]));
        let (_out_stream, output_audio) = crate::audio_io::setup_output_stream(true)?;
        let (_in_stream, input_audio) = crate::audio_io::setup_input_stream()?;
        let connector =
            native_tls::TlsConnector::builder().danger_accept_invalid_certs(true).build()?;
        let (stream, response) = tokio_tungstenite::connect_async_tls_with_config(
            uri,
            None,
            false,
            Some(tokio_tungstenite::Connector::NativeTls(connector)),
        )
        .await?;
        tracing::info!("connected, got {response:?}");

        initialize_panic_handler();
        startup()?;
        let mut t = Terminal::new(CrosstermBackend::new(std::io::stderr()))?;

        let (sender, mut receiver) = stream.split();
        let sender = Arc::new(tokio::sync::Mutex::new(MsgSender::new(sender)?));
        let (mut tx, rx) = tokio::io::duplex(100_000);

        let (action_tx, mut action_rx) = mpsc::unbounded_channel();
        let state = Arc::new(Mutex::new(State::Running));
        let stats = Arc::new(Mutex::new(Stats::new()));
        let mut app = App {
            should_quit: false,
            action_tx,
            ticker: 0,
            state: state.clone(),
            input_audio: input_audio.clone(),
            output_audio: output_audio.clone(),
            tui_log_state: tui_logger::TuiWidgetState::new(),
            subs: subs.clone(),
            stats: stats.clone(),
            current_db10: 0,
            sender: sender.clone(),
        };
        handle_event(app.action_tx.clone());

        tokio::spawn({
            let output_audio = output_audio.clone();
            async move {
                let mut decoder = opus::Decoder::new(24000, opus::Channels::Mono)?;
                let mut pr = ogg::reading::async_api::PacketReader::new(rx);
                let mut pcm_buf = vec![0f32; 24_000 * 120];
                let mut all_pcms = vec![];
                tracing::info!("waiting for audio data");
                while let Some(packet) = pr.next().await {
                    let packet = packet?;
                    if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags")
                    {
                        continue;
                    }
                    let size = decoder.decode_float(
                        &packet.data,
                        &mut pcm_buf,
                        /* Forward Error Correction */ false,
                    )?;
                    if size > 0 {
                        let pcm = &pcm_buf[..size];
                        all_pcms.push(pcm.to_vec());
                        // TODO: if the buffer is already containing more than x secs of audio, we
                        // should probably trim it.
                        output_audio.lock().unwrap().push_samples(pcm)?
                    }
                }
                let all_pcms = all_pcms.concat();
                tracing::info!(len = all_pcms.len(), "saving pcms with shape");
                let mut w = std::fs::File::create("received.wav")?;
                moshi::wav::write_pcm_as_wav(&mut w, &all_pcms, 24000)?;
                Ok::<(), anyhow::Error>(())
            }
        });
        tokio::spawn(async move {
            loop {
                let input = input_audio.lock().unwrap().take_all();
                if sender.lock().await.send_pcm(&input).await.is_err() {
                    break;
                };
                tokio::time::sleep(std::time::Duration::from_millis(20)).await
            }
        });
        tokio::spawn(async move {
            while let Some(received) = receiver.next().await {
                match received? {
                    Message::Close(_) => break,
                    Message::Text(text) => {
                        tracing::error!("unexpected text message {text}");
                        continue;
                    }
                    Message::Frame(_) | Message::Ping(_) | Message::Pong(_) => continue,
                    Message::Binary(bin) => {
                        if bin.is_empty() {
                            continue;
                        }
                        match bin[0] {
                            // Handshake
                            0 => {}
                            // Audio
                            1 => {
                                {
                                    let mut stats = stats.lock().unwrap();
                                    stats.recv_messages += 1;
                                    stats.recv_audio_messages += 1;
                                }
                                tx.write_all(&bin[1..]).await?;
                            }
                            2 => {
                                {
                                    let mut stats = stats.lock().unwrap();
                                    stats.recv_messages += 1;
                                    stats.recv_text_messages += 1;
                                }
                                let text = String::from_utf8_lossy(&bin[1..]).to_string();
                                subs.lock().unwrap().push(text)
                            }
                            3 => {
                                tracing::error!("unsupported control message")
                            }
                            4 => {
                                tracing::error!("unsupported metadata message")
                            }
                            mt => {
                                tracing::error!("unexpected message type {mt}");
                            }
                        }
                    }
                };
            }
            Ok::<_, anyhow::Error>(())
        });

        loop {
            t.draw(|f| {
                ui(f, &mut app);
            })?;
            if let Some(action) = action_rx.recv().await {
                update(&mut app, action).await?;
            }
            if app.should_quit {
                break;
            }
            app.ticker += 1;
        }

        shutdown()?;
        Ok(())
    }
}
