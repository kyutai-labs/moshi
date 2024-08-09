export const mimeTypeCheck = () => {
  const types = [
    "audio/ogg",
    "audio/wav",
    "audio/webm;codecs=opus",
    "audio/webm;codecs=pcm",
    "audio/webm;codecs=pcm_s16le",
    "audio/webm;codecs=pcm_f32le",
    "audio/mp3",
    "audio/aac",
    "audio/mp4",
    "audio/webm",
    "audio/mpeg",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  for (const mime of types) {
      console.log(mime, MediaRecorder.isTypeSupported(mime));
  }
}

const getVideoMimeType = () => {
  if (!MediaRecorder.isTypeSupported){
    return "video/mp4";
  }
  if (MediaRecorder.isTypeSupported("video/webm")) {
    return "video/webm";
  }
  if (MediaRecorder.isTypeSupported("video/mp4")) {
    return "video/mp4";
  }
  console.log("No supported video mime type found")
  return "";
};

const getAudioMimeType = () => {
  if (!MediaRecorder.isTypeSupported){
    return "audio/mp4";
  }
  if (MediaRecorder.isTypeSupported("audio/webm")) {
    return "audio/webm";
  }
  if (MediaRecorder.isTypeSupported("audio/mpeg")) {
    return "audio/mpeg";
  }``
  if (MediaRecorder.isTypeSupported("audio/mp4")) {
    return "audio/mp4";
  }
  console.log("No supported audio mime type found")
  return "";
}

export const getMimeType = (type: "audio" | "video") => {
  if(type === "audio") {
    return getAudioMimeType();
  }
  return getVideoMimeType();
}

export const getExtension = (type: "audio" | "video") => {
  if(getMimeType(type).includes("mp4")) {
    return "mp4";
  }
  if(getMimeType(type).includes("mpeg")) {
    return "mp3";
  }
  return "webm";
}