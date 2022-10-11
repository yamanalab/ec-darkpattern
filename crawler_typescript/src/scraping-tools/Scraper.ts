import Puppeteer from 'puppeteer';
import { PageSegmentBase } from './segment/PageSegmentBase';
import path from 'path';

export class Scraper {
  private pageSegment: PageSegmentBase;

  constructor(pageSegment: PageSegmentBase) {
    this.pageSegment = pageSegment;
  }

  public async scrapingAndSegmentPage(
    page: Puppeteer.Page,
    url: string,
    pathToScreenshotDir: string,
    waitTimeSec: number,
    viewport: Puppeteer.Viewport = { width: 1440, height: 900 }
  ): Promise<string[]> {
    await page.goto(url, {
      timeout: 0,
    });
    await page.setViewport(viewport);
    await page.waitForTimeout(waitTimeSec * 1000);
    const texts = await this.pageSegment.segmentPage(await page.$('html'));
    await this.takeScreenShot(
      page,
      path.join(pathToScreenshotDir, `whole-page.png`)
    );
    return texts;
  }

  private async takeScreenShot(
    page: Puppeteer.Page,
    pathToScreenshot: string
  ): Promise<void> {
    await page.screenshot({ path: pathToScreenshot, fullPage: true });
  }
}
