#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.help = false
params.manifest = null
params.ref = null
params.output_directory = null
params.kmer_size = 21
params.force = false
params.keep_intermediate = false
params.percent_identity = 90
params.image = null
params.seed = 42

// Failing because out of space - clear up some
// Remake pggb environment with latest pggb and vg=0.17.0

// fasta files for pggb don't seem large enough?

def invalidChars = [
    '(', ')', '|', '&', '<', '>', ';', ':', '"', "'", '`', '\\', '*', '?'
]

def usage = """Usage:
  nextflow run process_manifest.nf [options] --manifest MANIFEST 
  --working_directory WORKING_DIRECTORY

Sketch paired-end read datasets listed in a CSV manifest file, filtering out 
erroneous kmers based on estimated coverage.

Required arguments:
  --manifest MANIFEST  Path to CSV manifest in (name,read1,read2) format
  --ref REF            Path to reference sequence in fasta format
  --image IMAGE        Path to pggb singularity image
  --output_directory OUTPUT_DIRECTORY
                       Path to output directory

Options:
  --help               Show this help message and exit
  --force              Overwrite existing output directory
  --keep_intermediate  Keep intermediate files such as counts
"""

if (params.help) {
    log.info usage
    exit 0
}

if (!params.manifest) {
    error "Please provide --manifest with the path to the CSV file"
}
if (!params.ref) {
    error "Please provide --ref with the path to the reference"
}
if (!params.image) {
    error "Please provide --image with the path to the pggb singularity image"
}
if (!params.output_directory) {
    error "Please provide --output_directory with the path to the proposed output directory"
}

if (!file(params.ref).exists()) {
    error "Reference file does not exist: ${params.ref}"
}

def image_abs = file(params.image).toAbsolutePath()

process manysketch {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    path(manifest)

    output:
    path("manysketch.zip")

    script:
    """
    awk -F',' 'BEGIN { print "name,genome_filename,protein_filename" } { print \$1 "," \$2 "," "" }' "${manifest}" > manysketch_manifest.csv
    sourmash scripts manysketch --output=manysketch.zip --param-string='k=$params.kmer_size,scaled=1000,dna' manysketch_manifest.csv
    """
}

process estimate_ani {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    path(manysketch)

    output:
    path("distances.phylip")

    script:
    """
    sourmash2phylip \
        --output=distances.phylip \
        --kmer_length=$params.kmer_size ${manysketch}
    """
}

process embed {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    path(distances)

    output:
    path("embedding.tsv")

    script:
    """
    kpy-embed2 \
        --num_jobs=$task.cpus \
        --output_tsv=embedding.tsv \
        --seed=$params.seed \
        ${distances}
    """
}

process merge_clusters {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "signatures/*.sig.gz"
    publishDir "$params.output_directory", mode: 'copy', pattern: "clusters/*.csv"

    input:
    tuple path(zip), path(cluster)

    output:
    tuple path("clusters/*.csv"), path("signatures/*.sig.gz")

    script:
    """
    mergeClusters
    """
}

process make_fa {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*"

    input:
    path(cluster)

    output:
    path("graphs/*")

    script:
        """
    base=\$(basename "$cluster" .csv)
    n_haplotypes=\$(wc -l < "$cluster")
    fasta="graphs/\${base}.fa"
    echo \$fasta
    mkdir -p graphs
    awk -F',' '{
        name=\$1; file=\$2;
        idx=0;
        while ( getline seq < file ) {
            if ( seq ~ /^>/ ) {
                idx++;
                print ">" name "#" idx
            } else {
                print seq
            }
        }
        close(file)
    }' "$cluster" > "\$fasta"
    """
}

process assemble_gfa {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*"

    input:
    path(cluster)

    output:
    tuple path(cluster), path("graphs/*.gfa"), path("graphs/*")

    script:
    """
    base=\$(basename "$cluster" .csv)
    n_haplotypes=\$(wc -l < "$cluster")
    fasta="graphs/\${base}.fa"
    echo \$fasta
    mkdir -p graphs
    awk \
        -F ',' \
        -v ref_file="$params.ref" '
            BEGIN {
                idx=0;
                while ( ( getline seq < ref_file ) > 0 ) {
                    if ( seq ~ /^>/ ) {
                        idx++;
                        print ">" "ref" "#" idx
                    } else {
                        print seq
                    }
                }
                close(ref_file)
            }
            {
                name=\$1; file=\$2;
                idx=0;
                while ( ( getline seq < file ) > 0 ) {
                    if ( seq ~ /^>/ ) {
                        idx++;
                        print ">" name "#" idx
                    } else {
                        print seq
                    }
                }
                close(file)
            }
    ' "$cluster" > "\$fasta"
    samtools faidx "\$fasta"
    singularity \
        run \
        -B "\$(pwd):/data" \
        "$image_abs" \
        pggb \
            --input-fasta="/data/\${fasta}" \
            --n-haplotypes=\$((n_haplotypes + 1)) \
            --map-pct-id=$params.percent_identity \
            --mash-kmer=$params.kmer_size \
            --output-dir="/data/graphs" \
            --threads=11
    """
}

process autoindex {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*"
    // conda "vg==1.70"

    input:
    tuple path(cluster), path(gfa)

    output:
    path("graphs/*")

    script:
    """
    base=\$(basename "$cluster" .csv)
    mkdir -p graphs
    vg \
        autoindex \
        --workflow=giraffe \
        --gfa="$gfa" \
        --prefix="graphs/\${base}" \
        --threads=11
    """
}

// process autoindex {
//     cpus 11
//     publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*"
//     // conda "vg==1.70"

//     input:
//     tuple path(cluster), path(gfa)

//     output:
//     path("graphs/*")

//     script:
//     """
//     base=\$(basename "$cluster" .csv)
//     echo $cluster
//     echo $gfa
//     head $gfa
//     pwd
//     cp $gfa graph.gfa
//     singularity \
//         run \
//         -B "\$(pwd):/data" \
//         "$image_abs" \
//         ls -lha /data/
//     singularity \
//         run \
//         -B "\$(pwd):/data" \
//         "$image_abs" \
//         ls -lha /data/
//     singularity \
//         run \
//         -B "\$(pwd):/data" \
//         "$image_abs" \
//         head "/data/graph.gfa"
//     singularity \
//         run \
//         -B "\$(pwd):/data" \
//         "$image_abs" \
//         vg validate "/data/graph.gfa"
//     singularity \
//         run \
//         -B "\$(pwd):/data" \
//         "$image_abs" \
//         vg \
//             autoindex \
//             --workflow=giraffe \
//             --gfa="/data/graph.gfa" \
//             --prefix="/data/graphs/\${base}" \
//             --threads=11
//     """
// }

workflow {
    if (!workflow.resume) {
        def output_directory = file(params.output_directory)
        System.err.println("Initializing output directory: ${output_directory.toAbsolutePath()}")
        if (output_directory.exists()) {
            if (!output_directory.isDirectory()) {
                System.err.println("ERROR: existing file at output directory path")
                System.exit(1)
            }
            if (!params.force) {
                System.err.println("ERROR: output directory already exists (overwrite with --force)")
                System.exit(1)
            }
            System.err.println("Removing existing output directory (--force)")
            output_directory.deleteDir()
        }
        output_directory.mkdirs()
    }

    manysketch = manysketch(file(params.manifest))
    distances = estimate_ani(manysketch)
    embedding = embed(distances)
    clusters = merge_clusters(manysketch.combine(embedding))
    cluster_filenames = clusters
        .map { filenames, signatures -> filenames }
        .flatten()
    make_fa(cluster_filenames)
    cluster_filenames = cluster_filenames
        .toList()
        .map { it.reverse() }
        .flatten()
    graphs = assemble_gfa(cluster_filenames)
    gfas = graphs
        .filter { cluster, gfa, files -> !gfa.name.contains('empty') }
        .map { cluster, gfa, files -> tuple cluster, gfa }
    gfas.view()
    indices = autoindex(gfas)
}
