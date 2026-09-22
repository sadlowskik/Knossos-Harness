/* The plate Rome is drawn on: paper, water, and one island per mounted project.

   The geometry arrives from island.js already in percentage-of-canvas coordinates, which
   is the same coordinate system the districts, the settlements and the agent markers are
   positioned in — so the SVG uses `viewBox="0 0 100 100"` with `preserveAspectRatio="none"`
   and a district at 62% lands on the ground that island.js put at 62%. Strokes carry
   `vector-effect="non-scaling-stroke"` so that stretch never reaches a line weight.

   Every colour is a CSS custom property declared in styles/island.css. Nothing here
   computes geometry; it is a projection of what the generator already decided. */

const DEPTH_BANDS = ['band-far', 'band-near', 'band-shelf'];

function Island({ plate }) {
  const { island, heat } = plate;
  return <g className={`island${island.isle ? ' small-isle' : ''}`}>
    {/* Water first: the coast stroked wide and faint, three times, so the sea shallows
        toward the shore. The inner half of each band is covered by the land that follows. */}
    {DEPTH_BANDS.map((band) => <g key={band} className={`island-depth ${band}`}>
      {island.coast.map((d, index) => <path key={index} d={d} vectorEffect="non-scaling-stroke" />)}
    </g>)}
    {/* The lighter of the two coast lines, drawn under the land so only its outer half
        shows: a pale rule hugging the shore. */}
    <g className="island-shore-outer">
      {island.coast.map((d, index) => <path key={index} d={d} vectorEffect="non-scaling-stroke" />)}
    </g>

    <path className="island-land" d={island.land} />
    {/* Elevation: distance from the coast plus a second noise field, one value per cell,
        blurred into a gradient rather than a set of facets. */}
    <g className="island-relief" filter="url(#island-soften)">
      {island.tiles.map((tile, index) => (
        <path key={index} d={tile.d} style={{ opacity: (tile.elevation * 0.62).toFixed(3) }} />
      ))}
    </g>
    {/* One path per region, tinted by how long ago anyone worked there. */}
    <g className="island-tint">
      {island.regions.map((region) => (region.d
        ? <path key={region.key} className={`heat-${heat.get(region.key) ?? 'untouched'}`} d={region.d} />
        : null))}
    </g>
    <g className="island-borders">
      {island.borders.map((d, index) => <path key={index} d={d} vectorEffect="non-scaling-stroke" />)}
    </g>
    <g className="island-hatch">
      {island.hatches.map((d, index) => <path key={index} d={d} vectorEffect="non-scaling-stroke" />)}
    </g>
    <g className="island-shore">
      {island.coast.map((d, index) => <path key={index} d={d} vectorEffect="non-scaling-stroke" />)}
    </g>
  </g>;
}

export default function IslandPlate({ plates = [] }) {
  return <div className="island-ground" aria-hidden="true">
    <div className="island-paper" />
    <svg className="island-art" viewBox="0 0 100 100" preserveAspectRatio="none" focusable="false">
      <defs>
        <filter id="island-soften" x="-12%" y="-12%" width="124%" height="124%">
          <feGaussianBlur stdDeviation="0.9" />
        </filter>
      </defs>
      <rect className="island-sea" x="0" y="0" width="100" height="100" />
      {plates.map((plate) => <Island key={plate.key} plate={plate} />)}
    </svg>
    <div className="island-grain" />
    <div className="island-edge" />
  </div>;
}
