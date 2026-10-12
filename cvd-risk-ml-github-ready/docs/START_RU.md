# Как открыть проект и выложить на GitHub

1. Распакуй архив **в отдельную папку**, например `D:\учебка\АвтСхПр\data\cvd-risk-ml-github-ready`. В отличие от предыдущих обновлений, это **полный проект**, а не патч. Не нужно смешивать его со старыми файлами.
2. Из старого проекта скопируй `data/raw/CVD Dataset.csv` в новую папку `data/raw/`. **Не загружай CSV в GitHub**, он игнорируется через `.gitignore`. Токены Kaggle и `.env` также не загружай.
3. Открой терминал в новой папке и выполни:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows Git Bash
   python -m pip install -e ".[dev]"
   python -m pytest
   ```

4. Быстрый запуск (на Mac/Windows может занять несколько минут):

   ```bash
   python scripts/audit_dataset.py
   python scripts/train.py --feature-set all_clean --cv-only
   ```

   Для расширенной воспроизводимости:

   ```bash
   python scripts/ablate_preprocessing.py
   python scripts/permutation_importance_cv.py
   ```

   Последние две команды могут работать продолжительное время.
5. После проверки экспериментальных результатов можно сохранить финально обученную модель на **всей** размеченной выборке, без повторного использования теста:

   ```bash
   python scripts/fit_final.py
   ```

   Появится `artifacts/models/cvd_risk_all_clean_final.joblib` (не добавляется в GitHub). Скрипт не выводит метрики: обучение на всех данных не заменяет независимую проверку.
6. Создай пустой репозиторий на GitHub, например `cvd-risk-ml`. В Git Bash из папки проекта:

   ```bash
   git init
   git add .
   git status
   git commit -m "Build reproducible CVD classification study"
   git branch -M main
   git remote add origin https://github.com/USERNAME/cvd-risk-ml.git
   git push -u origin main
   ```

   Замени `USERNAME` на свой логин GitHub. Перед `git commit` проверь `git status`: `CVD Dataset.csv`, `kaggle.json`, `.venv/`, `.joblib` там **не должно быть**.

README рассчитан на техническое портфолио: не нужно прикладывать 87-страничный учебный отчёт. Итоги и ограничения описаны в `docs/experiments.md` и `docs/data_quality.md`. Не называй эту работу клинически валидированным прогнозом сердечно-сосудистых заболеваний.
