module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  parserOptions: {
    sourceType: "module",
    project: "./tsconfig.json"
  },
  plugins: ["@typescript-eslint", "import"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:import/typescript",
    "prettier"
  ],
  settings: {
    react: {
      version: "detect"
    }
  },
  env: {
    browser: true,
    es2022: true
  }
};
