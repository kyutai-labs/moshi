import { FC } from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement>; 
export const Button: FC<ButtonProps> = ({ children, className, ...props }) => {
  return (
    <button
      className={`border-2 disabled:bg-gray-100 border-white bg-black p-2 text-white hover:bg-gray-800 active:bg-gray-700  ${className ?? ""}`}
      {...props}
    >
      {children}
    </button>
  );
};
