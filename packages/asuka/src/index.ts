import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ASUKA_ROOT = dirname(dirname(dirname(__dirname)));

export { ASUKA_ROOT };
