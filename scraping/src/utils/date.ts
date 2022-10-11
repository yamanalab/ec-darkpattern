import { format } from 'date-fns';

export const formatDate = (date: Date, formatStr = 'yyyy-MM-dd'): string => {
  return format(date, formatStr);
};
