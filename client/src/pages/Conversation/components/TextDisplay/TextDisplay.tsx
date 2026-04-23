import { FC, useEffect, useRef } from "react";
import { useServerText } from "../../hooks/useServerText";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>; // kept in case you still need it
  displayColor: boolean | undefined;
};

const textDisplayColors = [
  "#c8ead9", "#F97316", "#1E3A8A", "#800080", "#FFC0CB", "#000000",
  "#f94144", "#f3722c", "#f9c74f", "#90be6d", "#577590", "#277da1",
  "#d19bf7", "#d7acf6", "#debdf5", "#e4cef4",
  "#ebe0f3", "#eef2f0", "#c8ead9", "#a4e2c4",
  "#80d9af", "#5bd09a", "#38c886"
];

function clamp_color(v: number) {
  return v <= 0
    ? 0
    : v >= textDisplayColors.length
      ? textDisplayColors.length - 1
      : v;
}

export const TextDisplay: FC<TextDisplayProps> = ({

}) => {
  const { text, textColor } = useServerText();
  const currentIndex = text.length - 1;

  // refs for each box
  const group0Ref = useRef<HTMLDivElement>(null);
  const group1Ref = useRef<HTMLDivElement>(null);

  // split into two groups
  const group0 = text
    .map((t, i) => ({ t, i }))
    .filter(({ i }) => textColor[i] === 0);

  const group1 = text
    .map((t, i) => ({ t, i }))
    .filter(({ i }) => textColor[i] === 1);

  // auto-scroll for group0
  useEffect(() => {
    if (group0Ref.current) {
      group0Ref.current.scrollTo({
        top: group0Ref.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [group0.length]);

  // auto-scroll for group1
  useEffect(() => {
    if (group1Ref.current) {
      group1Ref.current.scrollTo({
        top: group1Ref.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [group1.length]);

  return (
    <div className="h-full w-full flex flex-col gap-4 p-2">
      {/* First box */}
      <div
        ref={group0Ref}
        className="flex-1 overflow-auto p-2 border rounded text-white"
      >
        {group0.map(({ t, i }) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
            style={{ color: textDisplayColors[clamp_color(textColor[i])] }}
          >
            {t}
          </span>
        ))}
      </div>

      {/* Second box */}
      <div
        ref={group1Ref}
        className="flex-1 overflow-auto p-2 border rounded text-white"
      >
        {group1.map(({ t, i }) => (
          <span
            key={i}
            className={`${i === currentIndex ? "font-bold" : "font-normal"}`}
            style={{ color: textDisplayColors[clamp_color(textColor[i])] }}
          >
            {t}
          </span>
        ))}
      </div>
    </div>
  );
};
