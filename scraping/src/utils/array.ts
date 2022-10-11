export const shuffle = <T>(array: T[]): T[] => {
  const shuffleArray = array.slice();
  shuffleArray.sort(() => Math.random() - 0.5);
  return shuffleArray;
};

export const choice = <T>(array: T[], num: number, unique = true): T[] => {
  const choiceArray = [];
  const selected = new Set();
  while (choiceArray.length < num) {
    const idx = Math.floor(Math.random() * array.length);
    if (unique && selected.has(idx)) {
      continue;
    }
    choiceArray.push(array[idx]);
    selected.add(idx);
  }
  return choiceArray;
};

export const sliceIntoChunks = <T>(array: T[], chunkSize: number): T[][] => {
  const res: T[][] = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    const chunk = array.slice(i, i + chunkSize);
    res.push(chunk);
  }
  return res;
};
