import 'dotenv/config';

export const SLACK_WEBHOOK_URL = process.env.SLACK_WEBHOOK_URL as string;
export const HTTP_PROXY_URL = process.env.HTTP_PROXY_URL;
export const HTTP_PROXY_PORT =
  process.env.HTTP_PROXY_PORT && Number(process.env.HTTP_PROXY_PORT);
