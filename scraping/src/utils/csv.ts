import { parse } from 'csv/sync';
import fs from 'node:fs';

export const readCsv = (pathToCsv: string): string[][] => {
  const csvData = fs.readFileSync(pathToCsv, 'utf8');
  const records: string[][] = parse(csvData);
  return records;
};
