from distutils.core import setup

setup(
    name="modapt",
    packages=[
        "modapt",
        "modapt.dataset",
        "modapt/model",
    ],
    version="0.3",  # update for new ver
    license="MIT",
    description="Modular Domain Adaptation, https://github.com/jkvc/modapt",
    author="jkvc",
    author_email="kevinehc@gmail.com",
    url="https://github.com/jkvc/modapt",
    download_url="https://github.com/jkvc/modapt/archive/refs/tags/v_03.tar.gz",  # update for new ver
    keywords=["modapt", "modular", "domain", "adaptation"],
    install_requires=[
        "pandas",
        "torchvision",
        "nltk",
        "matplotlib",
        "transformers",
        "sklearn",
        "tensorboard",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
