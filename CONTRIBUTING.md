# Guía de contribución

Gracias por contribuir a este proyecto. Para mantener el historial del repositorio limpio y comprensible, seguimos las convenciones descritas a continuación.

## Flujo de trabajo

1. Crea una rama a partir de `main` siguiendo la convención de nombres (ver abajo).
2. Realiza tus cambios y haz commits siguiendo el estándar de Conventional Commits.
3. Sube la rama y abre un Pull Request hacia `main`.
4. Si el PR resuelve un issue, indícalo en la descripción con `Closes #<número>`.
5. Espera al menos una aprobación antes de fusionar (la rama `main` está protegida).

## Convención de nombres de ramas

Cada rama debe empezar con un prefijo que indique el tipo de cambio, seguido de una breve descripción en minúsculas separada por guiones:

| Prefijo     | Uso                                                         | Ejemplo                          |
|-------------|--------------------------------------------------------------|-----------------------------------|
| `feat/`     | Nueva funcionalidad                                          | `feat/entrenamiento-skipgram`     |
| `fix/`      | Corrección de errores                                         | `fix/calculo-softmax`             |
| `chore/`    | Tareas de mantenimiento, configuración, dependencias          | `chore/repo-conventions`          |
| `docs/`     | Cambios solo en documentación                                 | `docs/actualizar-readme`          |
| `refactor/` | Cambios internos de código que no alteran el comportamiento   | `refactor/optimizar-forward-pass` |
| `test/`     | Añadir o corregir tests                                       | `test/cobertura-negative-sampling`|
| `build/`    | Cambios en el sistema de build o dependencias (setup.py, requirements.txt) | `build/actualizar-numpy`  |
| `ci/`       | Cambios en la configuración de integración continua           | `ci/anadir-workflow-tests`        |

## Conventional Commits

Cada mensaje de commit debe seguir el formato:

```
<tipo>(<ámbito opcional>): <descripción breve en presente>
```

**Tipos permitidos:**

- `feat`: nueva funcionalidad para el usuario
- `fix`: corrección de un error
- `chore`: mantenimiento, configuración, tareas que no afectan al código fuente ni a los tests
- `docs`: cambios en documentación
- `refactor`: cambio de código que no corrige un bug ni añade una funcionalidad
- `test`: añadir o modificar tests
- `style`: cambios de formato (espacios, indentación) sin afectar la lógica
- `perf`: cambios que mejoran el rendimiento
- `build`: cambios que afectan al sistema de build o a dependencias externas (ej. `requirements.txt`, `setup.py`)
- `ci`: cambios en archivos y scripts de integración continua (ej. workflows de GitHub Actions)
- `revert`: revierte un commit anterior

**Sobre seguridad:** para issues o commits relacionados con seguridad (ej. vulnerabilidades en dependencias), lo más habitual en Conventional Commits es usar el tipo `fix` con el ámbito `security`, por ejemplo `fix(security): actualizar numpy por vulnerabilidad CVE-XXXX`. Si prefieres mantener `security` como categoría propia para etiquetar issues (label) o como tipo de commit personalizado, es válido siempre que lo uses de forma consistente en todo el repositorio.

**Ejemplos:**

```
feat: añadir entrenamiento skip-gram con negative sampling
fix(softmax): corregir desbordamiento numérico en la exponencial
chore: configurar protección de rama main y convenciones de repositorio
docs: documentar uso de la clase Word2Vec en el README
refactor(embeddings): simplificar inicialización de matrices de pesos
test: añadir tests para la función de similitud coseno
build: actualizar numpy a la versión 2.0
ci: añadir workflow de GitHub Actions para ejecutar tests
fix(security): actualizar dependencia con vulnerabilidad conocida
```

**Reglas adicionales:**

- Usa el imperativo/presente ("añadir", no "añadido" ni "añadiendo").
- La primera línea debe tener idealmente menos de 72 caracteres.
- Si el commit rompe compatibilidad con versiones anteriores, añade `!` después del tipo (ej. `feat!: cambiar firma de la función train`) y explica el cambio en el cuerpo del commit.
- Si necesitas más contexto, añade una línea en blanco después del título y describe el "por qué" del cambio.

## Pull Requests

- Usa un título claro, idealmente también en formato Conventional Commits.
- Describe brevemente qué cambia y por qué.
- Vincula el issue relacionado con `Closes #<número>` si corresponde.
- Asegúrate de que el código pasa los tests antes de solicitar revisión.
