# Contributing to rag-naive-2022

Thank you for your interest in contributing to rag-naive-2022! 🎉

## How to Contribute

1. **Fork** the repository at [github.com/mlvpatel/rag-naive-2022](https://github.com/mlvpatel/rag-naive-2022)
2. **Clone** your fork: `git clone https://github.com/<your-username>/rag-naive-2022.git`
3. Create a **feature branch**: `git checkout -b feature/amazing-feature`
4. **Commit** your changes: `git commit -m 'feat: add amazing feature'`
5. **Push** to your branch: `git push origin feature/amazing-feature`
6. Open a **Pull Request** against `main`

## Code Style

- Follow **PEP 8**
- Use **Black** for formatting: `make format`
- Run **flake8** + **isort** checks: `make lint`
- Write **docstrings** for all public functions and classes

## Testing

Run the full test suite before submitting a PR:

```bash
make test           # All tests with coverage report
make test-unit      # Unit tests only (fast, no infra needed)
```

Coverage must remain **≥ 70%**. New features should include corresponding tests.

## Environment Setup

```bash
make install        # Install all dependencies
cp .env.example .env
# Add at minimum: GOOGLE_API_KEY
```

## Questions?

Open an issue or contact: **@mlvpatel on GitHub**  
GitHub: [@mlvpatel](https://github.com/mlvpatel)
