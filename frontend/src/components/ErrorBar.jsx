/**
 * Error bar component — a fixed-position toast at the bottom of the screen.
 *
 * @param {{ message: string, visible: boolean }} props
 */
export default function ErrorBar({ message, visible }) {
  return (
    <div
      className={`error-bar${visible ? ' visible' : ''}`}
      id="errorBar"
    >
      {message}
    </div>
  );
}
