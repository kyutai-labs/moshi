import { FC } from "react";

type UserAudioStatsProps = {
  sentMessagesCount: number;
};

export const UserAudioStats: FC<UserAudioStatsProps> = ({
  sentMessagesCount,
}) => {
  return (
    <div>
      <h2 className="text-center text-lg">User Audio Stats</h2>
      <div>
        <div className="flex justify-between">
          <p className="text-md">Total messages:</p>
          <p>{sentMessagesCount}</p>
        </div>
      </div>
    </div>
  );
};
