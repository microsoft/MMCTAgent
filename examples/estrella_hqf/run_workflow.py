from hqf_spectral_step import HQFSpectralEmbeddingStep


def main():
    step = HQFSpectralEmbeddingStep(
        k=12,
        sigma=0.1,
        n_eigs=20,
        seed=0
    )

    out = step.run(level=5)

    print("\n=== HQF SPECTRAL WORKFLOW ===")
    print(f"Points:              {out['n_points']}")
    print(f"Classes:             {out['n_classes']}")
    print(f"Eigenpairs used:     {out['n_eigs']}")
    print(f"Fractal level:       {5}")
    print("")
    print(f"Accuracy:            {out['accuracy']:.4f}")
    print(f"F1 Macro:            {out['f1_macro']:.4f}")
    print(f"Balanced Accuracy:   {out['balanced_accuracy']:.4f}")
    print("")
    print(f"PCA Accuracy:        {out['accuracy_pca']:.4f}")
    print(f"PCA F1 Macro:        {out['f1_macro_pca']:.4f}")
    print("")
    print(f"Mean Spectral Gap:   {out['mean_gap']:.6f}")
    print(f"Gap Entropy:         {out['entropy_gap']:.6f}")


if __name__ == "__main__":
    main()
