/**
 * Generates assets/illustration.svg from the same geometry/animation logic as illustration.svelte.
 * Run: node scripts/generate-illustration.mjs
 */

import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
/**
 * Inline each logo as a self-contained data: URI.
 * The illustration is displayed through an <img> tag (and as an <object>
 * fallback), and browsers render SVGs loaded that way in "secure static mode",
 * which refuses to fetch external resources. A relative href like
 * "logos/foo.svg" therefore never loads. Embedding the logo as a data URI keeps
 * the illustration fully self-contained so it renders everywhere.
 */
const logoCache = new Map();
const logoHref = (icon) => {
	if (!logoCache.has(icon)) {
		const svg = readFileSync(join(__dirname, '..', 'assets', 'logos', icon), 'utf8');
		const b64 = Buffer.from(svg, 'utf8').toString('base64');
		logoCache.set(icon, `data:image/svg+xml;base64,${b64}`);
	}
	return logoCache.get(icon);
};

const R = 54;
const VB_W = 800;
const VB_H = 440;
const S3 = Math.sqrt(3);
const VB_X = -VB_W / 2;
const VB_Y = -VB_H / 2;

const THEMES = {
	light: {
		pattern: '#EDEFF2',
		text: '#1F2937',
		fade: 'white',
	},
	dark: {
		pattern: '#2D333B',
		text: '#E5E7EB',
		fade: '#111827',
	},
};

const ANIM_COLOR = '#FF2A4A';
const LINE_WIDTH = 1.0;
const DOT_RADIUS = 2.0;

const LOGO_GROUPS = [
	{ name: 'IE', icon: 'inference-endpoints.svg', text: null, tiles: [[0, 0]], iconSize: 70 },
	{
		name: 'SGLang',
		icon: 'engine-3.svg',
		text: 'SGL',
		tiles: [
			[-2, -2],
			[-1, -2],
		],
		iconSize: 30,
	},
	{
		name: 'vLLM',
		icon: 'engine-2.svg',
		text: 'vLLM',
		tiles: [
			[1, -2],
			[2, -2],
		],
		iconSize: 30,
	},
	{
		name: 'LlamaCpp',
		icon: 'engine-4.svg',
		text: 'LLaMA',
		tiles: [
			[-3, 0],
			[-2, 0],
		],
		iconSize: 30,
	},
	{
		name: 'TGI',
		icon: 'engine-5.svg',
		text: 'TEI',
		tiles: [
			[2, 0],
			[3, 0],
		],
		iconSize: 30,
	},
	{
		name: 'Transformers',
		icon: 'engine-1.svg',
		text: 'Transformers',
		tiles: [
			[-1, 2],
			[0, 2],
			[1, 2],
		],
		iconSize: 30,
	},
];

function tileCenter(i, j, R) {
	const xOffset = j % 2 !== 0 ? (R * S3) / 2 : 0;
	return [i * R * S3 + xOffset, j * 1.5 * R];
}

function hexVerts(cx, cy, R) {
	return [
		[cx, cy - R],
		[cx + (R * S3) / 2, cy - R / 2],
		[cx + (R * S3) / 2, cy + R / 2],
		[cx, cy + R],
		[cx - (R * S3) / 2, cy + R / 2],
		[cx - (R * S3) / 2, cy - R / 2],
	];
}

const TAKEN = new Set();
for (const g of LOGO_GROUPS) for (const [i, j] of g.tiles) TAKEN.add(`${i},${j}`);

const J_MAX = Math.floor(VB_H / 2 / (1.5 * R)) + 2;
const I_MAX = Math.floor(VB_W / 2 / (R * S3)) + 2;

function buildGrid() {
	const hexPaths = [];
	const dots = [];
	for (let j = -J_MAX; j <= J_MAX; j++) {
		for (let i = -I_MAX; i <= I_MAX; i++) {
			if (TAKEN.has(`${i},${j}`)) continue;
			const [cx, cy] = tileCenter(i, j, R);
			const v = hexVerts(cx, cy, R);
			let d = `M ${v[0][0].toFixed(2)} ${v[0][1].toFixed(2)}`;
			for (let k = 1; k < 6; k++) d += ` L ${v[k][0].toFixed(2)} ${v[k][1].toFixed(2)}`;
			d += ' Z';
			d += ` M ${cx.toFixed(2)} ${cy.toFixed(2)} L ${v[0][0].toFixed(2)} ${v[0][1].toFixed(2)}`;
			d += ` M ${cx.toFixed(2)} ${cy.toFixed(2)} L ${v[2][0].toFixed(2)} ${v[2][1].toFixed(2)}`;
			d += ` M ${cx.toFixed(2)} ${cy.toFixed(2)} L ${v[4][0].toFixed(2)} ${v[4][1].toFixed(2)}`;
			hexPaths.push(d);
			for (const [vx, vy] of [...v, [cx, cy]]) {
				dots.push({ cx: vx, cy: vy });
			}
		}
	}
	return { d: hexPaths.join(' '), dots };
}

const FONT_SIZE = 24;
const TEXT_GAP = 6;
const ICON_Y_ADJUST = -2;
const MONO_RATIO = 0.6;

function buildLogos() {
	const out = [];
	for (const g of LOGO_GROUPS) {
		const centers = g.tiles.map(([i, j]) => tileCenter(i, j, R));
		const cx = centers.reduce((s, c) => s + c[0], 0) / centers.length;
		const cy = centers.reduce((s, c) => s + c[1], 0) / centers.length;
		let iconLeft, textX;
		if (g.text) {
			const tw = g.text.length * FONT_SIZE * MONO_RATIO;
			const comboW = g.iconSize + TEXT_GAP + tw;
			iconLeft = cx - comboW / 2;
			textX = iconLeft + g.iconSize + TEXT_GAP;
		} else {
			iconLeft = cx - g.iconSize / 2;
		}
		const nudge = g.text ? ICON_Y_ADJUST : 0;
		out.push({
			icon: g.icon,
			iconLeft,
			iconY: cy - g.iconSize / 2 + nudge,
			iconSize: g.iconSize,
			text: g.text,
			textX,
			textY: cy,
		});
	}
	return out;
}

function enginePaths() {
	return [
		[
			[(-R * S3) / 2, (-5 * R) / 2],
			[(-R * S3) / 2, (-3 * R) / 2],
			[0, -R],
		],
		[
			[-R * S3, -2 * R],
			[-R * S3, -R],
			[(-R * S3) / 2, -R / 2],
		],
		[
			[(R * S3) / 2, (-5 * R) / 2],
			[(R * S3) / 2, (-3 * R) / 2],
			[0, -R],
		],
		[
			[R * S3, -2 * R],
			[R * S3, -R],
			[(R * S3) / 2, -R / 2],
		],
		[
			[(-3 * R * S3) / 2, -R / 2],
			[-R * S3, -R],
			[(-R * S3) / 2, -R / 2],
		],
		[
			[(-3 * R * S3) / 2, R / 2],
			[-R * S3, 0],
			[(-R * S3) / 2, R / 2],
		],
		[
			[(3 * R * S3) / 2, -R / 2],
			[R * S3, -R],
			[(R * S3) / 2, -R / 2],
		],
		[
			[(3 * R * S3) / 2, R / 2],
			[R * S3, 0],
			[(R * S3) / 2, R / 2],
		],
		[
			[-R * S3, 2 * R],
			[(-R * S3) / 2, (3 * R) / 2],
			[(-R * S3) / 2, R / 2],
		],
		[
			[0, 2 * R],
			[0, R],
		],
		[
			[R * S3, 2 * R],
			[(R * S3) / 2, (3 * R) / 2],
			[(R * S3) / 2, R / 2],
		],
	];
}

const TIME_SLOT = [1, 7, 4, 10, 5, 0, 8, 2, 9, 3, 6];

function buildAnimations() {
	const CYCLE = 14;
	const SPEED = 40;
	const TRAIL = 39;
	const LINE_THICK = 1.6;
	const LINE_OPACITY = 0.6;
	const FLASH_HALF = 0.012;
	const BIG_GAP = 99999;

	const paths = enginePaths();
	const N = paths.length;
	const stagger = CYCLE / N;

	const lines = [];
	const flashes = [];

	for (let idx = 0; idx < paths.length; idx++) {
		const waypoints = paths[idx];
		const begin = TIME_SLOT[idx] * stagger;
		const segLengths = [];
		for (let k = 0; k < waypoints.length - 1; k++) {
			const a = waypoints[k];
			const b = waypoints[k + 1];
			segLengths.push(Math.hypot(b[0] - a[0], b[1] - a[1]));
		}
		const totalLen = segLengths.reduce((a, b) => a + b, 0);
		const L = Math.min(TRAIL, totalLen * 0.6);

		let tGrow = L / SPEED;
		let tTravel = (totalLen - L) / SPEED;
		let tShrink = L / SPEED;
		let tTotal = tGrow + tTravel + tShrink;
		if (tTotal > CYCLE) {
			const scale = (CYCLE / tTotal) * 0.95;
			tGrow *= scale;
			tTravel *= scale;
			tShrink *= scale;
			tTotal = tGrow + tTravel + tShrink;
		}
		const fGrow = tGrow / CYCLE;
		const fTravel = (tGrow + tTravel) / CYCLE;
		const fEnd = tTotal / CYCLE;

		const d =
			'M ' +
			waypoints.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join(' L ');

		const daKt = [0, fGrow, fTravel, fEnd, 1];
		const daVs = [
			`0,${BIG_GAP}`,
			`${L.toFixed(2)},${BIG_GAP}`,
			`${L.toFixed(2)},${BIG_GAP}`,
			`0,${BIG_GAP}`,
			`0,${BIG_GAP}`,
		];
		const doKt = [0, fGrow, fTravel, fEnd, 1];
		const doVs = [0, 0, -(totalLen - L), -totalLen, -totalLen];

		lines.push({
			d,
			beginSec: begin,
			cycleSec: CYCLE,
			daKt: daKt.map((x) => x.toFixed(4)).join('; '),
			daVs: daVs.join('; '),
			doKt: doKt.map((x) => x.toFixed(4)).join('; '),
			doVs: doVs.map((v) => v.toFixed(3)).join('; '),
			thickness: LINE_THICK,
			opacity: LINE_OPACITY,
		});

		const waypointPos = [0];
		let cum = 0;
		for (const sl of segLengths) {
			cum += sl;
			waypointPos.push(cum);
		}
		for (let w = 0; w < waypoints.length; w++) {
			const [vx, vy] = waypoints[w];
			const p = waypointPos[w];
			const tHead = p <= L ? p / SPEED : tGrow + (p - L) / SPEED;
			let tTail;
			if (p === 0) tTail = tGrow;
			else if (p >= totalLen - L) tTail = tGrow + tTravel + (p - (totalLen - L)) / SPEED;
			else tTail = tGrow + p / SPEED;
			let onStart = Math.min(tHead, tTotal) / CYCLE;
			let onEnd = Math.min(tTail, tTotal) / CYCLE;
			if (onEnd <= onStart) onEnd = Math.min(onStart + FLASH_HALF, 1);
			const fadeW = Math.min(FLASH_HALF, Math.max(0.001, (onEnd - onStart) / 4));
			const pre = Math.max(0, onStart - fadeW);
			const post = Math.min(1, onEnd + fadeW);
			let kt, vals;
			if (onStart <= 0) {
				kt = [0, onEnd, post, 1];
				vals = [1, 1, 0, 0];
			} else if (post >= 1) {
				kt = [0, pre, onStart, 1];
				vals = [0, 0, 1, 1];
			} else {
				kt = [0, pre, onStart, onEnd, post, 1];
				vals = [0, 0, 1, 1, 0, 0];
			}
			flashes.push({
				cx: vx,
				cy: vy,
				beginSec: begin,
				cycleSec: CYCLE,
				kt: kt.map((x) => x.toFixed(4)).join('; '),
				vals: vals.join('; '),
			});
		}
	}
	return { lines, flashes };
}

const { d: gridPath, dots } = buildGrid();
const logos = buildLogos();
const { lines, flashes } = buildAnimations();

function esc(s) {
	return String(s)
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;');
}

/** Parse "39,99999" from the grow-phase dasharray keyframe. */
function staticDasharray(daVs) {
	const grow = daVs.split('; ')[1];
	return grow ?? '0,99999';
}

function renderSvg(theme, { animated = true } = {}) {
	const PATTERN_COLOR = theme.pattern;
	const TEXT_COLOR = theme.text;
	const FADE_COLOR = theme.fade;

	const parts = [];
parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
parts.push(
	`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="${VB_X} ${VB_Y} ${VB_W} ${VB_H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Inference Endpoints engine illustration">`,
);
parts.push('<defs>');
parts.push('<linearGradient id="ep-fade-t" x1="0" y1="0" x2="0" y2="1">');
parts.push(`<stop offset="0" stop-color="${FADE_COLOR}" stop-opacity="1" />`);
parts.push(`<stop offset="1" stop-color="${FADE_COLOR}" stop-opacity="0" />`);
parts.push('</linearGradient>');
parts.push('<linearGradient id="ep-fade-b" x1="0" y1="0" x2="0" y2="1">');
parts.push(`<stop offset="0" stop-color="${FADE_COLOR}" stop-opacity="0" />`);
parts.push(`<stop offset="1" stop-color="${FADE_COLOR}" stop-opacity="1" />`);
parts.push('</linearGradient>');
parts.push('<linearGradient id="ep-fade-l" x1="0" y1="0" x2="1" y2="0">');
parts.push(`<stop offset="0" stop-color="${FADE_COLOR}" stop-opacity="1" />`);
parts.push(`<stop offset="1" stop-color="${FADE_COLOR}" stop-opacity="0" />`);
parts.push('</linearGradient>');
parts.push('<linearGradient id="ep-fade-r" x1="0" y1="0" x2="1" y2="0">');
parts.push(`<stop offset="0" stop-color="${FADE_COLOR}" stop-opacity="0" />`);
parts.push(`<stop offset="1" stop-color="${FADE_COLOR}" stop-opacity="1" />`);
parts.push('</linearGradient>');
parts.push('</defs>');

parts.push(
	`<g fill="none" stroke="${PATTERN_COLOR}" stroke-width="${LINE_WIDTH}"><path d="${esc(gridPath)}" /></g>`,
);
parts.push(`<g fill="${PATTERN_COLOR}">`);
for (const d of dots) {
	parts.push(`<circle cx="${d.cx}" cy="${d.cy}" r="${DOT_RADIUS}" />`);
}
parts.push('</g>');

for (const l of logos) {
	parts.push(
		`<image href="${logoHref(l.icon)}" x="${l.iconLeft}" y="${l.iconY}" width="${l.iconSize}" height="${l.iconSize}" preserveAspectRatio="xMidYMid meet" />`,
	);
	if (l.text) {
		parts.push(
			`<text x="${l.textX}" y="${l.textY}" font-family="'IBM Plex Mono', monospace" font-weight="400" font-size="${FONT_SIZE}" fill="${TEXT_COLOR}" dominant-baseline="middle" text-anchor="start">${esc(l.text)}</text>`,
		);
	}
}

for (const ln of lines) {
	if (animated) {
		parts.push(
			`<path d="${esc(ln.d)}" stroke="${ANIM_COLOR}" stroke-opacity="${ln.opacity}" stroke-width="${ln.thickness}" fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="0,99999" stroke-dashoffset="0">` +
				`<animate attributeName="stroke-dasharray" values="${ln.daVs}" keyTimes="${ln.daKt}" dur="${ln.cycleSec}s" begin="${ln.beginSec}s" repeatCount="indefinite" calcMode="linear" />` +
				`<animate attributeName="stroke-dashoffset" values="${ln.doVs}" keyTimes="${ln.doKt}" dur="${ln.cycleSec}s" begin="${ln.beginSec}s" repeatCount="indefinite" calcMode="linear" />` +
				`</path>`,
		);
	} else {
		parts.push(
			`<path d="${esc(ln.d)}" stroke="${ANIM_COLOR}" stroke-opacity="${ln.opacity}" stroke-width="${ln.thickness}" fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="${staticDasharray(ln.daVs)}" stroke-dashoffset="0" />`,
		);
	}
}

if (animated) {
	for (const f of flashes) {
		parts.push(
			`<circle cx="${f.cx}" cy="${f.cy}" r="${DOT_RADIUS}" fill="${ANIM_COLOR}" opacity="0">` +
				`<animate attributeName="opacity" values="${f.vals}" keyTimes="${f.kt}" dur="${f.cycleSec}s" begin="${f.beginSec}s" repeatCount="indefinite" />` +
				`</circle>`,
		);
	}
}

parts.push(`<rect x="${VB_X}" y="${VB_Y}" width="${VB_W}" height="50" fill="url(#ep-fade-t)" />`);
parts.push(
	`<rect x="${VB_X}" y="${VB_Y + VB_H - 50}" width="${VB_W}" height="50" fill="url(#ep-fade-b)" />`,
);
parts.push(`<rect x="${VB_X}" y="${VB_Y}" width="100" height="${VB_H}" fill="url(#ep-fade-l)" />`);
parts.push(
	`<rect x="${VB_X + VB_W - 100}" y="${VB_Y}" width="100" height="${VB_H}" fill="url(#ep-fade-r)" />`,
);
parts.push('</svg>');
return parts.join('\n');
}

const outDir = join(__dirname, '..', 'assets');
mkdirSync(outDir, { recursive: true });

for (const [name, theme] of Object.entries(THEMES)) {
	const suffix = name === 'light' ? '' : `-${name}`;
	for (const [variant, animated] of [
		['', true],
		['-static', false],
	]) {
		const outPath = join(outDir, `illustration${suffix}${variant}.svg`);
		const svg = renderSvg(theme, { animated });
		writeFileSync(outPath, svg, 'utf8');
		console.log(`Wrote ${outPath} (${svg.length} bytes)`);
	}
}
