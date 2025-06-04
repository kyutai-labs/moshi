// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::protocol::MsgType;
use anyhow::Result;
use axum::extract::ws;
use candle::{Device, IndexOp, Tensor};
use std::sync::Arc;

use kaudio::ogg_opus;

struct Sender {
    tx: tokio::sync::broadcast::Sender<ws::Message>,
    encoder: kaudio::ogg_opus::Encoder,
}

impl Sender {
    fn send_raw(&mut self, data: &[u8]) -> Result<()> {
        let msg = ws::Message::Binary(data.to_vec().into());
        let _ = self.tx.send(msg);
        Ok(())
    }

    fn send_pcm(&mut self, pcm: &[f32]) -> Result<()> {
        let data = self.encoder.encode_page(pcm)?;
        let msg: Vec<u8> = [&[MsgType::Audio.to_u8()], data.as_slice()].concat();
        let msg = ws::Message::Binary(msg.into());
        // We do not fail on send errors as these mean that there is no subscribers though
        // new subscribers may show up later.
        let _ = self.tx.send(msg);
        Ok(())
    }

    fn send_ping(&mut self) {
        let msg = ws::Message::Binary(vec![MsgType::Ping.to_u8()].into());
        let _ = self.tx.send(msg);
    }
}

struct Room {
    sender: Arc<tokio::sync::Mutex<Sender>>,
    header_message: ws::Message,
    rx: tokio::sync::broadcast::Receiver<ws::Message>,
}

impl Room {
    fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::broadcast::channel(10);
        let encoder = ogg_opus::Encoder::new(24_000)?;
        let header_message: Vec<u8> = [&[MsgType::Audio.to_u8()], encoder.header_data()].concat();
        let header_message = ws::Message::Binary(header_message.into());
        let sender = Sender { tx, encoder };
        let sender = Arc::new(tokio::sync::Mutex::new(sender));
        tokio::spawn({
            let sender = sender.clone();
            async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    let mut sender = sender.lock().await;
                    sender.send_ping();
                }
            }
        });
        Ok(Self { sender, header_message, rx })
    }
}

pub struct Mimi {
    audio_tokenizer: moshi::mimi::Mimi,
    device: Device,
    #[allow(unused)]
    instance_name: String,
    auth_recv: bool,
    #[allow(unused)]
    log_dir: std::path::PathBuf,
    rooms: std::collections::HashMap<String, Room>,
    default_room: Option<String>,
}

impl Mimi {
    pub fn new(mimi: &crate::MimiConfig, config: &crate::Config, dev: &Device) -> Result<Self> {
        let audio_tokenizer = moshi::mimi::load(&mimi.audio_tokenizer_file, Some(8), dev)?;
        let mut rooms = std::collections::HashMap::new();
        for room in mimi.rooms.iter() {
            rooms.insert(room.to_string(), Room::new()?);
        }

        Ok(Self {
            audio_tokenizer,
            device: dev.clone(),
            log_dir: config.log_dir.clone().into(),
            instance_name: config.instance_name.clone(),
            auth_recv: mimi.auth_recv,
            default_room: mimi.default_room.clone(),
            rooms,
        })
    }

    pub fn auth_recv(&self) -> bool {
        self.auth_recv
    }

    pub async fn recv_socket(&self, socket: ws::WebSocket, room_id: Option<String>) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};

        let room_id = match (room_id, self.default_room.as_ref()) {
            (Some(r), _) => r,
            (None, Some(d)) => d.to_string(),
            (None, None) => anyhow::bail!("no room_id provided"),
        };
        let room = match self.rooms.get(&room_id) {
            None => anyhow::bail!("unknown room"),
            Some(room) => room,
        };

        // Re-subscribe early to have more chances to have a message immediately available.
        let mut rx = room.rx.resubscribe();
        let (mut ws_sender, mut ws_receiver) = socket.split();
        let recv_loop = async move { while ws_receiver.next().await.is_some() {} };
        let mut handshake = vec![MsgType::Handshake.to_u8()];
        handshake.resize(9, 0u8);
        if let Err(err) = ws_sender.send(ws::Message::binary(handshake)).await {
            tracing::error!("error sending header {err:?}");
            return Ok(());
        }
        if let Err(err) = ws_sender.send(room.header_message.clone()).await {
            tracing::error!("error sending header {err:?}");
            return Ok(());
        }
        let send_loop = async move {
            loop {
                let msg = match rx.recv().await {
                    Ok(msg) => msg,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                        continue;
                    }
                };
                if let Err(err) = ws_sender.send(msg).await {
                    tracing::error!("exiting recv loop, error in send: {err:?}");
                    break;
                }
            }
        };
        tokio::select! {
            _ = send_loop => tracing::info!("recv_socket: send loop exited"),
            _ = recv_loop => tracing::info!("recv_socket: receiver disconnected"),
        }
        Ok(())
    }

    pub async fn send_socket(&self, socket: ws::WebSocket, room_id: String) -> Result<()> {
        use futures_util::StreamExt;

        tracing::info!("connected to sender for {room_id}");
        let room = match self.rooms.get(&room_id) {
            None => anyhow::bail!("unknown room"),
            Some(room) => room,
        };
        let mut sender = match room.sender.try_lock() {
            Ok(s) => s,
            Err(_) => anyhow::bail!("already a producer"),
        };
        let (_ws_sender, mut ws_receiver) = socket.split();
        let mut audio_tokenizer = self.audio_tokenizer.clone();

        let mut pcm_all = vec![];
        while let Some(msg) = ws_receiver.next().await {
            let msg = match msg? {
                ws::Message::Binary(b) => b.to_vec(),
                _ => continue,
            };
            if msg.is_empty() {
                continue;
            }
            match MsgType::from_u8(msg[0]) {
                Ok(MsgType::Text) => {
                    // Forward directly the text messages.
                    sender.send_raw(&msg)?;
                }
                Ok(MsgType::Codes) => {
                    let codes: Vec<u32> = msg[1..]
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    let ncodes = codes.len();
                    let codes = Tensor::from_vec(codes, (1, ncodes, 1), &self.device)?;
                    let pcm = audio_tokenizer.decode_step(&codes.into(), &().into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                        for v in pcm.into_iter() {
                            pcm_all.push(v);
                            if pcm_all.len() == 1920 {
                                sender.send_pcm(&pcm_all)?;
                                pcm_all.clear();
                            }
                        }
                        // Sleep to avoid starving the scheduler.
                        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                    }
                }
                t => {
                    tracing::warn!("unexpected msg type {t:?}");
                    continue;
                }
            }
        }
        tracing::info!("send_socket: exiting send loop");

        Ok(())
    }
}
