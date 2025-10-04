import React from "react";
import ReactDOM from "react-dom/client";

const App: React.FC = () => (
  <main>
    <h1>Reality's Ledger UI</h1>
    <p>Operator dashboard placeholder.</p>
  </main>
);

const root = document.getElementById("root");
if (root) {
  const app = ReactDOM.createRoot(root);
  app.render(<App />);
}

export default App;
