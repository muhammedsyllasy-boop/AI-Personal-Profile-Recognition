"""Generate 20,000 synthetic student/professional profiles for career-domain prediction."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

def generate_dataset(n_samples=20000, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 46, n_samples)
    education = rng.choice(["High School", "Diploma", "Bachelor", "Master", "PhD"], n_samples,
                           p=[.12, .16, .40, .25, .07])
    experience = np.clip(rng.gamma(2.0, 2.0, n_samples), 0, 25).round(1)
    programming = rng.integers(0, 101, n_samples)
    communication = rng.integers(0, 101, n_samples)
    creativity = rng.integers(0, 101, n_samples)
    analytical = rng.integers(0, 101, n_samples)
    leadership = rng.integers(0, 101, n_samples)
    design = rng.integers(0, 101, n_samples)
    business = rng.integers(0, 101, n_samples)
    science = rng.integers(0, 101, n_samples)

    # Scores are intentionally overlapping to make this a realistic demonstration dataset.
    scores = np.column_stack([
        .34*programming + .28*analytical + .15*science + .10*experience + rng.normal(0, 10, n_samples),
        .30*communication + .25*leadership + .25*business + .10*experience + rng.normal(0, 10, n_samples),
        .35*creativity + .35*design + .15*communication + rng.normal(0, 10, n_samples),
        .34*science + .30*analytical + .16*programming + rng.normal(0, 10, n_samples),
        .32*communication + .28*science + .20*analytical + rng.normal(0, 10, n_samples),
    ])
    domains = np.array(["Technology", "Business & Management", "Creative Design",
                        "Science & Research", "Healthcare"])
    # Add label noise so the exercise is not perfectly deterministic.
    labels = domains[scores.argmax(axis=1)]
    noisy = rng.random(n_samples) < 0.07
    labels[noisy] = rng.choice(domains, noisy.sum())

    df = pd.DataFrame({
        "age": age, "education_level": education, "experience_years": experience,
        "programming_score": programming, "communication_score": communication,
        "creativity_score": creativity, "analytical_score": analytical,
        "leadership_score": leadership, "design_score": design,
        "business_score": business, "science_score": science,
        "professional_domain": labels
    })
    path = DATA_DIR / "synthetic_profiles.csv"
    df.to_csv(path, index=False)
    print(f"Created {len(df):,} profiles: {path}")
    return df

if __name__ == "__main__":
    generate_dataset()
