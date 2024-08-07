type InputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  error?: string;
}

export const Input = ({className, error, ...props}:InputProps) => {
  return (
    <div className="pb-8 relative">
      <input
        {...props}
        className={`border-2 disabled:bg-gray-100 border-white bg-black p-2 outline-none text-white hover:bg-gray-800 focus:bg-gray-800 p-2 ${className ?? ""}`}
      />
      {error && <p className=" absolute text-red-800">{error}</p>}
    </div>
  );
}