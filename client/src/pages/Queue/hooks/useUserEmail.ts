import { useCallback, useState } from "react";
import {z} from "zod";

const validateEmail = z.string().email();

export const useUserEmail = (isBypass: boolean) => {
  const [userEmail, setUserEmail] = useState<string>('');
  const [error, setError] = useState<string|null>(null);

  const validate = useCallback((email: string) => {
      if(isBypass) {
        setError(null);
        return true;
      }
      const result = validateEmail.safeParse(email);
      if(result.success) {
        setError(null);
        return true;
      }
      setError('Invalid email address');
      return false;
  }, [setError]);
  return {userEmail, setUserEmail, error, validate};
}
