from setuptools import find_namespace_packages, find_packages, setup


packages = sorted(
    set(
        find_packages(include=["multiflow*", "colabdesign*"])
        + find_namespace_packages(include=["openfold*", "ProteinMPNN*"])
        + ["openfold", "multiflow", "ProteinMPNN", "colabdesign"]
    )
)


setup(
    name="chamaileon",
    version="0.1.0",
    description="Open-source Chamaileon codebase built on the multiflow layout.",
    packages=packages,
    include_package_data=True,
    package_data={
        "openfold": ["resources/*.txt"],
        "multiflow": ["configs/*.yaml", "configs/*.json"],
        "ProteinMPNN": [
            "ca_model_weights/*.pt",
            "vanilla_model_weights/*.pt",
            "helper_scripts/*.py",
            "helper_scripts/*.sh",
            "helper_scripts/other_tools/*.py",
        ],
        "colabdesign": [
            "af/weights/*.npy",
            "mpnn/weights/*.pkl",
            "mpnn/weights_soluble/*.pkl",
            "rf/*.css",
            "rf/*.js",
        ],
    },
)
