import Puppeteer from 'puppeteer';

export interface PageSegmentBase {
  segmentPage(element: Puppeteer.ElementHandle | null): Promise<string[]>;
}
