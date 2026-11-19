# Changelog

Todos los cambios notables del proyecto **Word2VecWithNumpy** se documentarán en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere al [Versionado Semántico](https://semver.org/lang/es/).

## [Unreleased]

## [0.1.0] - 2026-11-19
### Added
- Primera versión funcional del motor Word2Vec desde cero.
- Implementación algorítmica de la arquitectura CBOW con Negative Sampling usando NumPy puro.
- Script de volcado de corpus masivo desde Wikipedia (`corpusextract.py`).
- Módulos de inicialización, *forward pass*, cálculo de gradientes y *backward pass* optimizados con indexación de tensores.
- Batería de tests inicial para verificar la integridad de las funciones matemáticas.
- Confirmación de reproducibilidad y portabilidad del entorno en múltiples dispositivos Windows.
- Documentación principal (`README.md`) con la arquitectura del sistema y los pasos de instalación.
- Creación del `CHANGELOG.md`.
- Guía de contribución (`CONTRIBUTING.md`) con el flujo de trabajo y:
    - Convención de nombres de ramas
    - Conventional Commits
    - Reglas para Pull Requests
- Carpeta `.github` con los siguientes archivos:
    - `CODEOWNERS`
    - `ISSUE_TEMPLATE/general.md`
    - `PULL_REQUEST_TEMPLATE.md`

[Unreleased]: https://github.com/AlfonsoJimena/Word2vecWithNumpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AlfonsoJimena/Word2vecWithNumpy/releases/tag/v0.1.0
