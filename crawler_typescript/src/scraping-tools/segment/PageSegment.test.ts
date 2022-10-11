import { PageSegment } from './PageSegment';
import Puppeteer from 'puppeteer';

let browser: Puppeteer.Browser;
let page: Puppeteer.Page;

describe('', () => {
  beforeEach(async () => {
    browser = await Puppeteer.launch();
    page = await browser.newPage();
  });

  afterEach(async () => {
    await browser.close();
  });
  it('segmentPage("<html><body><p>hoge</p></body></html>") should return ["hoge"]', async (): Promise<void> => {
    const html = '<html><body><p>hoge</p></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual(['hoge']);
  });
  it('segmentPage("<html><body><div><p>hoge</p><p>fuga</p></div></body></html>") should return ["hoge","fuga"]', async (): Promise<void> => {
    const html = '<html><body><div><p>hoge</p><p>fuga</p></div></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual(['hoge', 'fuga']);
  });
  it('segmentPage("<html><body><div><p>hoge</p><div><p>fuga</p></div></div></body></html>") should return ["hoge","fuga"]', async (): Promise<void> => {
    const html =
      '<html><body><div><p>hoge</p><div><p>fuga</p></div></div></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual(['hoge', 'fuga']);
  });
  it('segmentPage("<html><body><p>hoge<span>fuga</span></p></body></html>") should return ["hogefuga"]', async (): Promise<void> => {
    const html = '<html><body><p>hoge<span>fuga</span></p></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual(['hogefuga']);
  });
  it('segmentPage("<html><body><div><p>hoge</p><span>fuga</span></div></body></html>") should return ["hoge","fuga"]', async (): Promise<void> => {
    const html =
      '<html><body><div><p>hoge</p><span>fuga</span></div></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual(['hoge', 'fuga']);
  });
  it('segmentPage("<html><body><script>console.log("fuga")</script></body></html>") should return []', async (): Promise<void> => {
    const html =
      '<html><body><script>console.log("fuga")</script></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual([]);
  });
  it('segmentPage("<p>hoge<script>console.log("hoge")</script></p>") should return ["hoge"]', async (): Promise<void> => {
    const html =
      '<html><body><p>hoge<script>console.log("hoge")</script></p></body></html>';
    await page.setContent(html);
    const aruneshPageSegment = new PageSegment(page);
    const texts = await aruneshPageSegment.segmentPage();
    expect(texts).toEqual(['hoge']);
  });
});
