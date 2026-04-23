import { FC, useEffect, useMemo, useRef } from "react";
import { useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
  displayColor: boolean | undefined;
};

// Palette 2: Purple to Green Moshi
// sns.diverging_palette(288, 145, s=90, l=72, n=11)
const textDisplayColors = [
  "#d19bf7", "#d7acf6", "#debdf5", "#e4cef4",
  "#ebe0f3", "#eef2f0", "#c8ead9", "#a4e2c4",
  "#80d9af", "#5bd09a", "#38c886",
];

function clamp_color(v: number) {
  if (v <= 0) return 0;
  if (v >= textDisplayColors.length) return textDisplayColors.length - 1;
  return v;
}

const STREAM_LABELS: Record<number, string> = {
  0: "Assistant",
  1: "User",
};

const STREAM_DEFAULT_COLORS = ["#ffffff", "#5bd09a", "#d19bf7", "#f9c74f"];

const StreamBox: FC<{
  streamIdx: number;
  label: string;
  entries: { t: string; i: number; color: number }[];
  currentIndex: number;
  displayColor: boolean;
}> = ({ streamIdx, label, entries, currentIndex, displayColor }) => {
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (boxRef.current) {
      boxRef.current.scrollTo({ top: boxRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [entries.length]);
  const defaultColor =
    STREAM_DEFAULT_COLORS[streamIdx] ?? STREAM_DEFAULT_COLORS[STREAM_DEFAULT_COLORS.length - 1];
  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="text-xs text-gray-400 px-2 py-1">{label}</div>
      <div
        ref={boxRef}
        className="flex-1 overflow-auto p-2 border border-white/20 rounded"
        style={{ color: defaultColor }}
      >
        {entries.map(({ t, i, color }) => (
          <span
            key={i}
            className={i === currentIndex ? "font-bold" : "font-normal"}
            style={displayColor ? { color: textDisplayColors[clamp_color(color)] } : undefined}
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
};

export const TextDisplay: FC<TextDisplayProps> = ({ containerRef, displayColor }) => {
  const { text, textColor } = useServerText();
  const currentIndex = text.length - 1;

  const groupedEntries = useMemo(() => {
    const groups = new Map<number, { t: string; i: number; color: number }[]>();
    text.forEach((t, i) => {
      const color = textColor[i] ?? 0;
      if (!groups.has(color)) groups.set(color, []);
      groups.get(color)!.push({ t, i, color });
    });
    return groups;
  }, [text, textColor]);

  const streamIndices = Array.from(groupedEntries.keys()).sort((a, b) => a - b);

  const prevScrollTop = useRef(0);
  useEffect(() => {
    if (containerRef.current) {
      prevScrollTop.current = containerRef.current.scrollTop;
      containerRef.current.scroll({ top: containerRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [text, containerRef]);

  if (streamIndices.length <= 1) {
    return (
      <div className="h-full w-full max-w-full max-h-full p-2 text-white">
        {text.map((t, i) => (
          <span
            key={i}
            className={i === currentIndex ? "font-bold" : "font-normal"}
            style={
              displayColor && textColor.length === text.length
                ? { color: textDisplayColors[clamp_color(textColor[i])] }
                : undefined
            }
          >
            {t}
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="h-full w-full flex flex-col gap-2 p-2">
      {streamIndices.map(streamIdx => (
        <StreamBox
          key={streamIdx}
          streamIdx={streamIdx}
          label={STREAM_LABELS[streamIdx] ?? `Stream ${streamIdx}`}
          entries={groupedEntries.get(streamIdx)!}
          currentIndex={currentIndex}
          displayColor={!!displayColor}
        />
      ))}
    </div>
  );
};
