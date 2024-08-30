from __future__ import annotations

import sys
import torch
import os

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))

sys.path.append(os.path.join(os.path.dirname(cwd), "msh"))

import time
import asyncio
import logging

from concurrent.futures import ThreadPoolExecutor
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe, utils

from dotenv import load_dotenv

import hf_models_downloader
import model_state

load_dotenv()


logger = logging.getLogger("moshi-agent")
logger.setLevel(logging.DEBUG)

moshi_model: model_state.ModelState


async def entrypoint(ctx: JobContext):
    audio_task: asyncio.Task | None = None

    speech_q = asyncio.Queue()
    loop = asyncio.get_running_loop()

    executor = ThreadPoolExecutor(max_workers=1)

    @utils.log_exceptions(logger=logger)
    async def _audio_stream_task(
        participant: rtc.RemoteParticipant, track: rtc.RemoteAudioTrack
    ):
        audio_stream = rtc.AudioStream(
            track,
            sample_rate=model_state.SAMPLE_RATE,
            num_channels=model_state.NUM_CHANNELS,
        )

        moshi_model.reset()
        async for ev in audio_stream:
            frames = await loop.run_in_executor(executor, moshi_model, ev.frame)

            for f in frames:
                speech_q.put_nowait(f)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        nonlocal audio_task
        if isinstance(track, rtc.RemoteAudioTrack):
            audio_task = asyncio.create_task(_audio_stream_task(participant, track))

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    source = rtc.AudioSource(model_state.SAMPLE_RATE, model_state.NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("audio-output", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    publication = await ctx.room.local_participant.publish_track(track, options)
    await publication.wait_for_subscription()

    # capture frame to the source from speech_q
    while True:
        frame = await speech_q.get()
        await source.capture_frame(frame)


if __name__ == "__main__":
    logger.info("starting moshi-agent")
    hf_models_downloader.download_moshi_models(
        path="/moshi_models",
        force=False,
    )

    with torch.no_grad():
        moshi_model = model_state.ModelState.load()
        moshi_model.warmup()

        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, multi_process_mode=False))
