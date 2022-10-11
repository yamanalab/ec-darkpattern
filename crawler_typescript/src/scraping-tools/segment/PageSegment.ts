import Puppeteer from 'puppeteer';
import { PageSegmentBase } from './PageSegmentBase';
import { BLOCK_ELEMENTS, INLINE_ELEMENT } from '../../const/html/element-type';
import { NODE_TYPES } from '../../const/html/node-type';

export class PageSegment implements PageSegmentBase {
  private page: Puppeteer.Page;
  private IGNORE_ELEMENTS = ['SCRIPT', 'STYLE', 'NOSCRIPT', 'BR', 'HR'];

  constructor(page: Puppeteer.Page) {
    this.page = page;
  }

  public async segmentPage(): Promise<string[]> {
    return this.segmentElement(await this.page.$('body'));
  }

  private async segmentElement(
    element: Puppeteer.ElementHandle | null
  ): Promise<string[]> {
    if (!element) {
      return [];
    }
    if (!(await this.hasChildNodesIgnoreElement(element))) {
      if (await this.isAllChildNodesTextNodeOrInlineElement(element)) {
        const text = await this.getTextContent(element);
        if (text) {
          return [text];
        }
      }
    }

    const childNodes = await this.getChildNodes(element);
    let texts: string[] = [];
    for (const child of childNodes) {
      const nodeType = await this.getNodeType(child);
      if (nodeType == NODE_TYPES.TEXT_NODE) {
        const textContent = await this.getTextContent(child);
        if (textContent) {
          texts.push(textContent);
        }
      }

      const tagName = await this.getTagName(child);
      if (!tagName || this.IGNORE_ELEMENTS.includes(tagName.toUpperCase())) {
        continue;
      }
      if (tagName.toUpperCase() in BLOCK_ELEMENTS) {
        texts = texts.concat(await this.segmentElement(child));
      }
      if (tagName.toUpperCase() in INLINE_ELEMENT) {
        const textContent = await this.getTextContent(child);
        if (textContent) {
          texts.push(textContent);
        }
      }
    }
    return texts;
  }
  private async hasChildNodesIgnoreElement(
    element: Puppeteer.ElementHandle
  ): Promise<boolean> {
    const childNodes = await this.getChildNodes(element);
    for (const child of childNodes) {
      const tagName = await this.getTagName(child);
      if (tagName == null) {
        continue;
      }
      if (this.IGNORE_ELEMENTS.includes(tagName.toUpperCase())) {
        return true;
      }
    }
    return false;
  }

  private async isAllChildNodesTextNodeOrInlineElement(
    element: Puppeteer.ElementHandle
  ): Promise<boolean> {
    const childNodes = await this.getChildNodes(element);
    for (const child of childNodes) {
      const tagName = await this.getTagName(child);
      const nodeType = await this.getNodeType(child);
      if (tagName == null) {
        if (nodeType !== NODE_TYPES.TEXT_NODE) {
          return false;
        }
      } else if (
        !(nodeType == NODE_TYPES.TEXT_NODE || tagName in INLINE_ELEMENT)
      ) {
        return false;
      }
    }
    return true;
  }
  /**
   * return child nodes of element
   */
  private async getChildNodes(
    element: Puppeteer.ElementHandle
  ): Promise<(Puppeteer.ElementHandle | null)[]> {
    const listHandle = await this.page.evaluateHandle((e) => {
      return e.childNodes;
    }, element);
    const properties = await listHandle.getProperties();
    const childNodes: (Puppeteer.ElementHandle | null)[] = [];
    for (const property of properties.values()) {
      const element = property.asElement();
      childNodes.push(element);
    }
    return childNodes;
  }

  private async getTextContent(
    element: Puppeteer.ElementHandle | null
  ): Promise<string | undefined> {
    const text = await (await element?.getProperty('textContent'))?.jsonValue();
    return text as string | undefined;
  }
  private async getTagName(
    element: Puppeteer.ElementHandle | null
  ): Promise<string | undefined> {
    return await (await element?.getProperty('tagName'))?.jsonValue();
  }
  private async getNodeType(
    element: Puppeteer.ElementHandle | null
  ): Promise<number | undefined> {
    return await (await element?.getProperty('nodeType'))?.jsonValue();
  }
}
