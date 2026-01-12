#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.help = false
params.manifest = null
params.output_directory = null
params.kmer_size = 21
params.force = false
params.keep_intermediate = false

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
if (!params.output_directory) {
    error "Please provide --output_directory with the path to the proposed output directory"
}

// process sketch {
//     cpus 1
//     publishDir "$params.output_directory", mode: 'copy', pattern: "*"

//     input:
//     tuple val(name), path(fa)

//     output:
//     path("${name}.sig"), optional: true

//     script:
//     """
//     echo "test" $fa
//     sourmash sketch dna -p 'k=$params.kmer_size,scaled=1000' "${fa}" -o"${name}.sig"
//     """
// }

// process convert_manifest {
//     input: path(manifest)

//     output: path("manifest.csv")

//     script:
//     """
//     """
// }

process manysketch {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    path(manifest)

    output:
    path("manysketch.zip")

    script:
    """
    x="test"
    echo \$x > test.txt
    awk -F',' 'BEGIN { print "name,genome_filename,protein_filename" } { print \$1 "," \$2 "," "" }' "${manifest}" > manysketch_manifest.csv
    sourmash scripts manysketch --output=manysketch.zip --param-string='k=$params.kmer_size,scaled=1000,dna' manysketch_manifest.csv
    """
}

// process manysketch {
//     cpus 11
//     publishDir "$params.output_directory", mode: 'copy', pattern: "*"

//     input:
//     path(manifest)

//     output:
//     // path("manysketch.zip")
//     tuple path("manysketch.zip"), path("test.txt")

//     script:
//     """
//     x="test"
//     echo \$x > test.txt
//     awk -F',' 'BEGIN { print "name,genome_filename,protein_filename" } { print \$1 "," \$2 "," "" }' "${manifest}" > manysketch_manifest.csv
//     sourmash scripts manysketch --output=manysketch.zip --param-string='k=$params.kmer_size,scaled=1000,dna' manysketch_manifest.csv
//     """
// }

process estimate_ani {
    cpus 11
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    path(manysketch)

    output:
    path("distances.phylip")

    script:
    """
    sourmash2phylip --output=distances.phylip --kmer_length=$params.kmer_size ${manysketch}
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
    kpy-embed2 --num_jobs=$task.cpus --output_tsv=embedding.tsv ${distances}
    """
}

// process unpack_clusters {
//     cpus 11

//     input:
//     tuple path(zip), path(embedding)

//     output:
//     path("*.cluster.txt")

//     script:
//     """
//     sourmash2clusters 
//     """
// }

// process merge_sketches {
//     cpus 1
//     publishDir "$params.output_directory", mode: 'copy', pattern: "*"

//     input:
//     tuple path(zip), path(cluster)

//     output:
//     path("*.cluster.sig.gz")

//     script:
//     """
//     cluster2sketch
//     """
// }

process merge_clusters {
    cpus 11
    // publishDir "$params.output_directory/clusters", mode: 'copy', pattern: "cluster/*.sig.gz"
    publishDir "$params.output_directory", mode: 'copy', pattern: "signatures/*.sig.gz"
    publishDir "$params.output_directory", mode: 'copy', pattern: "clusters/*.csv"
    // publishDir "$params.output_directory", mode: 'copy', pattern: ["signatures/*.sig.gz", "clusters/*.csv"]

    input:
    tuple path(zip), path(cluster)

    output:
    // tuple path("*.cluster.csv"), path("*.cluster.sig.gz")
    // tuple path("*.cluster.csv"), path("signatures/*.sig.gz")
    tuple path("clusters/*.csv"), path("signatures/*.sig.gz")
    // path("clusters/*.csv")

    script:
    """
    mergeClusters
    """
}

// process assemble_gfa {
//     cpus 1
//     publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*.gfa"
//     // publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*.fa"

//     input:
//     path(cluster)

//     output:
//     path("graphs/*.fa")

//     script:
//     """
//     base=\$(basename "$cluster" .csv)
//     mkdir -p graphs
//     cut -d',' -f2 "$cluster" | xargs cat > "graphs/\${base}.fa"
//     AlfaPang "graphs/\${base}.fa" "graphs/\${base}.gfa" $params.kmer_size
//     """
//     // AlfaPang sequences.fa graphs/${cluster}.gfa $params.kmer_size
//     // xargs cat < $cluster > "graphs/$(basename clusters/mycluster.csv .csv).fa"
//     // xargs cat < "$cluster" > "graphs/\${base}.fa"
// }

process assemble_gfa {
    cpus 1
    // publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*.gfa"
    publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*"

    input:
    path(cluster)

    output:
    tuple path("graphs/*.gfa"), path("graphs/*.num")

    script:
    """
    base=\$(basename "$cluster" .csv)
    mkdir -p graphs
    wc -l < "$cluster" > "graphs/\${base}.num"
    cut -d',' -f2 "$cluster" | xargs cat > "graphs/\${base}.fa"
    AlfaPang "graphs/\${base}.fa" "graphs/\${base}.gfa" $params.kmer_size
    """
}

process smooth_gfa {
    cpus 11
    // publishDir "$params.output_directory", mode: 'copy', pattern: "*.list"
    publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*.gfa.smoothed"

    input:
    tuple path(graph), val(num)

    output:
    // path("*.list")
    path("graphs/*.gfa.smoothed")

    script:
    // """
    // ls -lh > "${graph}.list"
    // """
    """
    mkdir -p graphs
    smoothxg -t${task.cpus} -r${num} -g"${graph}" -o"graphs/${graph}.smoothed"
    """
}

process fix_gfa {
    cpus 11
    // publishDir "$params.output_directory", mode: 'copy', pattern: "*.list"
    publishDir "$params.output_directory", mode: 'copy', pattern: "graphs/*.gfa.smoothed.fixed"

    input:
    path(graph)

    output:
    // path("*.list")
    path("graphs/*.gfa.smoothed.fixed")

    script:
    // """
    // ls -lh > "${graph}.list"
    // """
    """
    mkdir -p graphs
    gfaffix -p${task.cpus} -o"graphs/${graph}.fixed" "${graph}"
    """
}

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

    // manifest = convert_manifest(file(params.manifest))
    // manysketch = manysketch(manifest)
    manysketch = manysketch(file(params.manifest))
    distances = estimate_ani(manysketch)
    embedding = embed(distances)
    // clusters = unpack_clusters(sketches.combine(embedding)).flatten()
    // merge_sketches(sketches.combine(clusters))
    clusters = merge_clusters(manysketch.combine(embedding))
    cluster_filenames = clusters
        .map { filenames, signatures -> filenames }
        .flatten()
        // .buffer(size: 10)
        // .last()
        // .flatten()
    // clusters.view()
    // cluster_filenames.view()
    // assemble_gfa(clusters.flatten())
    raw_gfas = assemble_gfa(cluster_filenames)
    raw_gfas2 = raw_gfas
        .map { graph, num -> tuple(graph, num.text.trim().toInteger()) }
    // raw_gfas2.view()
    smoothed_gfas = smooth_gfa(raw_gfas2)
    fixed_gfas = fix_gfa(smoothed_gfas)

    // zip_files = Channel.fromPath('clusters/*.zip')
    // zip_files = Channel.fromPath('*.zip')
    // zip_files.view()

    // test(zip_files)
}

    // lines = Channel
    //     .fromPath(params.manifest)
    //     .splitText()
    //     .filter { it.trim() }

    // processedInput = lines.map { line ->
    //     def p = line.trim().split(',')
    //     if (p.size() != 2)
    //         error "Invalid line in CSV: '${line}'"
    //     def name = p[0]
    //     if (invalidChars.any { c -> name.contains(c) }) {
    //         error "Name contains invalid characters ${invalidChars.join('')}:\n${name}"
    //     }
    //     def fa = p[1]
    //     if (invalidChars.any { c -> fa.contains(c) }) {
    //         error "Filename contains invalid characters ${invalidChars.join('')}:\n${fa}"
    //     }
    //     tuple(name, fa)
    // }

    // manysketch(processedInput.collect())

    // sketches = sketch(processedInput)
// }
