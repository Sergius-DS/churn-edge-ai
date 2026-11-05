/* tslint:disable */
/* eslint-disable */

/**
 * 🚀 FUNCIÓN DE INICIALIZACIÓN (Llamar UNA SOLA VEZ desde JS al cargar la página)
 * Parsea el modelo y el contrato y los guarda en memoria caché.
 */
export function init_engine(model_json: string, contract_json: string): void;

/**
 * 🚀 FUNCIÓN DE INFERENCIA (Ultra-rápida, sin parsing de modelo)
 * ✅ Ya no recibe model_json ni contract_json. Usa el caché global.
 */
export function predict_churn(input_json: string, consent: boolean): string;

/**
 * Función auxiliar para validar paridad con Python (usa el caché).
 */
export function predict_from_scaled_features(features_json: string): string;

/**
 * Función de verificación de paridad (usa el caché).
 */
export function verify_parity(features_json: string, expected_proba: number, tolerance: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly init_engine: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly predict_churn: (a: number, b: number, c: number, d: number) => void;
    readonly predict_from_scaled_features: (a: number, b: number, c: number) => void;
    readonly verify_parity: (a: number, b: number, c: number, d: number, e: number) => void;
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
