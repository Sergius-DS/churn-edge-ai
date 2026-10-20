/* tslint:disable */
/* eslint-disable */

/**
 * Función principal exportada a JavaScript.
 *
 * # Argumentos
 * - `input_json`: JSON string con los datos crudos del cliente
 * - `model_json`: JSON string del modelo XGBoost (xgb_model_edge.json)
 * - `contract_json`: JSON string del contrato (inference_contract_edge.json)
 *
 * # Retorna
 * JSON string con la predicción.
 */
export function predict_churn(input_json: string, model_json: string, contract_json: string): string;

/**
 * Función auxiliar para validar paridad con Python.
 * Toma features ya escaladas y retorna probabilidad + margin.
 */
export function predict_from_scaled_features(features_json: string, model_json: string): string;

/**
 * Función de verificación de paridad.
 * Compara la predicción WASM con el expected de Python.
 */
export function verify_parity(features_json: string, expected_proba: number, model_json: string, tolerance: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly predict_churn: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly predict_from_scaled_features: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly verify_parity: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
