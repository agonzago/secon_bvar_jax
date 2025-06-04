from setuptools import setup, find_packages

setup(
    name="clean_gpm_bvar_trends",
    version="0.1.0",
    description="A package for GPM BVAR trend analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name", # Replace with actual author if known, otherwise leave as placeholder
    author_email="your.email@example.com", # Replace with actual email if known
    url="https://github.com/yourusername/yourrepository", # Replace with actual URL if known
    packages=find_packages(include=['clean_gpm_bvar_trends', 'clean_gpm_bvar_trends.*']),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "jax",
        "jaxlib", # jaxlib is often a separate requirement for jax
        "numpyro",
        "yax", # As specified by the user
        "arviz", # Often used with numpyro for results analysis & plotting
        "sympy" # Added missing dependency
    ],
    python_requires=">=3.8", # Specify a reasonable Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, will create LICENSE file later
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    package_data={
        'clean_gpm_bvar_trends': ['models/*.gpm'], # To include GPM files once moved
    },
    include_package_data=True,
)
