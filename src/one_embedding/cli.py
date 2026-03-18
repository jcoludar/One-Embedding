"""One Embedding CLI."""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import click
from src.one_embedding import __version__


@click.group()
@click.version_option(__version__, prog_name="oneemb")
def main():
    """One Embedding — universal protein embedding codec."""
    pass


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
@click.option("-m", "--model", default="prot_t5", help="PLM: prot_t5, esm2")
def extract(input, output, model):
    """Sequence FASTA → per-residue embeddings."""
    from src.one_embedding import embed as _embed
    output = output or input.rsplit(".", 1)[0] + ".h5"
    click.echo(f"Extracting {model} embeddings...")
    _embed(input, output, model=model)
    click.echo(f"Saved: {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
def encode(input, output):
    """Per-residue H5 → compressed .oemb."""
    from src.one_embedding import encode as _encode
    output = output or input.rsplit(".", 1)[0] + ".oemb"
    click.echo("Encoding...")
    _encode(input, output)
    click.echo(f"Saved: {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
def inspect(input):
    """Show .oemb file contents."""
    from src.one_embedding.io import inspect_oemb
    info = inspect_oemb(input)
    for k, v in info.items():
        click.echo(f"  {k}: {v}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
def disorder(input):
    """Predict intrinsic disorder."""
    from src.one_embedding.tools.disorder import predict
    results = predict(input)
    for pid, scores in results.items():
        n_dis = (scores > 0.5).sum()
        click.echo(f"{pid}: {len(scores)} residues, {n_dis} disordered ({n_dis/len(scores):.0%})")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("--db", type=click.Path(exists=True), default=None, help="Reference database .oemb")
@click.option("-k", default=5, help="Number of neighbors")
def search(input, db, k):
    """Find structural neighbors."""
    from src.one_embedding.tools.search import find_neighbors
    results = find_neighbors(input, db=db or input, k=k)
    for pid, hits in results.items():
        click.echo(f"\n{pid}:")
        for h in hits:
            click.echo(f"  {h['name']:20s} sim={h['similarity']:.4f}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("protein_a")
@click.argument("protein_b")
def align(input, protein_a, protein_b):
    """Align two proteins from .oemb file."""
    from src.one_embedding.tools.align import align_pair
    r = align_pair(input, protein_a, protein_b)
    click.echo(f"Score: {r['score']:.2f}, Aligned: {r['n_aligned']}")


if __name__ == "__main__":
    main()
