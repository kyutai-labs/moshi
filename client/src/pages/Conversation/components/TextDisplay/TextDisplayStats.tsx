import { FC } from "react";

type TextDisplayStatsProps = {
  totalTextMessages: number;
};
export const TextDisplayStats: FC<TextDisplayStatsProps> = ({
  totalTextMessages,
}) => {
  return (
    <div className="w-60 flex-shrink-0">
      <h2 className="text-center text-lg">Text Display Stats</h2>
      <div>
        <div className="flex justify-evenly">
          <p className="text-md">Total messages:</p>
          <p>{totalTextMessages}</p>
        </div>
      </div>
    </div>
  );
};
