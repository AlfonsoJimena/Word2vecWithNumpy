# Changelog

Todos los cambios notables del proyecto **Word2VecWithNumpy** se documentarán en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere al [Versionado Semántico](https://semver.org/lang/es/).

## [1.0.0] - 2026-08-14
### Added
- Confirmación de reproducibilidad y portabilidad del entorno en múltiples dispositivos Windows.
- Refactorización de la documentación principal (README.md) para reflejar con mayor precisión la arquitectura del sistema y los pasos de instalación.
- Creación del CHANGELOG.md

## [0.9.0] - 2026-08-13
### Added
- Primera versión funcional (Beta) del motor Word2Vec desde cero.
- Implementación algorítmica de la arquitectura CBOW con Negative Sampling usando NumPy puro.
- Script de volcado de corpus masivo desde Wikipedia (`corpusextract.py`).
- Módulos de inicialización, *forward pass*, cálculo de gradientes y *backward pass* optimizados con indexación de tensores.
- Batería de tests inicial para verificar la integridad de las funciones matemáticas.