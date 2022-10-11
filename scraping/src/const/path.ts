import path from 'path';

export const PROJECT_ROOT = path.join(__dirname, '../../');

export const DATASET_DIR = path.join(PROJECT_ROOT, 'dataset');

export const DARKPATTERN_CSV_PATH = path.join(
  DATASET_DIR,
  'darkpatterns',
  'darkpattern.csv'
);

export const DARKPATTERN_CSV_DIR = path.join(DATASET_DIR, 'darkpatterns');

export const DARKPATTERN_SPLITED_CSV_DIR = path.join(
  DATASET_DIR,
  'darkpatterns',
  'splited'
);

export const CRAWLED_DATA_DIR = path.join(PROJECT_ROOT, 'crawled-data');
export const CRAWLED_TSV_DIR = path.join(DATASET_DIR, 'crawled-tsv');
