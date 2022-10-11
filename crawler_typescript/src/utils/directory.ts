import fs from 'node:fs';

export const createDirIfNotExist = (dir: string): void => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }
};
