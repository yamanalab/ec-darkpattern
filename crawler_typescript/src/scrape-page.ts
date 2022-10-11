import puppeteer from 'puppeteer';
import { Scraper } from './scraping-tools/Scraper';
import { PageSegment } from './scraping-tools/segment/PageSegment';
import { DARKPATTERN_CSV_PATH, CRAWLED_DATA_DIR } from './const/path';

import path from 'node:path';
import { formatDate } from './utils/date';
import { readCsv } from './utils/csv';
import { sliceIntoChunks } from './utils/array';
import { createDirIfNotExist } from './utils/directory';
import fs from 'node:fs';

const scrapeAndSegmentPage = async (
  url: string,
  pageId: string,
  dirToSave: string
): Promise<string[]> => {
  // Initialzie Scraper.
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  const scraper = new Scraper(new PageSegment(page));

  console.log(`start scraping ${url}`);

  // Execute Scraping & Segmentation.
  let texts: string[] = [];
  try {
    texts = await scraper.scrapingAndSegmentPage(page, url, dirToSave, 5);
  } catch (e) {
    console.log('scraping failed: ', url);
    console.log('Error: ', e);
  }
  // Filtering Crawled Texts.
  const continuousSpaceOverTwoCharactorRule = /\s{2,}/g;
  const textsReplaced = texts
    .map((text) => text.replace(continuousSpaceOverTwoCharactorRule, ' ')) // Replace Continuous Spaces to a Space. Eg. "    " → " "
    .map((text) => text.replace('\n', '')); // Replace New Line("\n") to a Space. Eg. "\n" → " "

  // Close(Destroy) Browser.
  await browser.close();

  // Convert Crawled Texts to TSV Format.
  const tsvHeader = ['page_id', 'text', 'url'].join('\t');
  const tsvBody = textsReplaced
    .map((text) => `${pageId}\t${text}\t${url}`)
    .join('\n');
  const tsvData = `${tsvHeader}\n${tsvBody}`;

  // Save TSV.
  const pathToTsv = path.join(dirToSave, 'page-text.tsv');
  fs.writeFileSync(pathToTsv, tsvData);

  console.log(`${pageId} ${url} is scraped`);

  return textsReplaced;
};

const scrapeMathursPages = async (pathToCsv: string) => {
  const records = readCsv(pathToCsv);
  const recordBody = records.slice(1);
  const urls = recordBody.map((record) => {
    const URL_COLUMN_IDX = 7;
    return record[URL_COLUMN_IDX];
  });

  const todayStr = formatDate(new Date());
  const dirToSave = path.join(CRAWLED_DATA_DIR, todayStr);

  createDirIfNotExist(dirToSave);

  const chunkSize = 5; // = number of threads.
  const urlChunks = sliceIntoChunks(urls, chunkSize);

  for (let chunkIdx = 0; chunkIdx < urlChunks.length; chunkIdx++) {
    const urlChunk = urlChunks[chunkIdx];
    const promises = urlChunk.map((url, idx) => {
      const pageId = String(chunkIdx * chunkSize + idx + 1);
      const dirToSavePerPage = path.join(dirToSave, pageId);
      createDirIfNotExist(dirToSavePerPage);
      return scrapeAndSegmentPage(url, pageId, dirToSavePerPage);
    });
    await Promise.all(promises);
  }
  console.log('scraping is finished 😆');
};

const main = () => {
  scrapeMathursPages(DARKPATTERN_CSV_PATH);
};

main();
