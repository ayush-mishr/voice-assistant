/**
 * Orb visualizer component.
 * Renders the orb circle and its two pulse-ring elements.
 * The actual animations are driven by CSS classes on the parent #app container.
 */
export default function Orb() {
  return (
    <div className="orb-container">
      <div className="orb" id="orb"></div>
      <div className="orb-pulse-ring" id="pulseRing1"></div>
      <div className="orb-pulse-ring orb-pulse-ring-2" id="pulseRing2"></div>
    </div>
  );
}
