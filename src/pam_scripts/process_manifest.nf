#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.help = false
params.manifest = null
params.output_directory = null
params.kmer_size = 21
params.low = 0.1
params.high = 5.0
params.counter = "kmc"
params.force = false
params.keep_intermediate = false

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
  --kmer_size KMER_SIZE
                       kmer size to use
                         (default: $params.kmer_size)
  --low LOW            Lower threshold as proportion of coverage 
                         (default: $params.low)
  --high HIGH          Upper threshold as proportion of coverage
                         (default: $params.high)
  --counter COUNTER    Kmer counting software to use
                         (default: $params.counter)
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

process count_kmc {
    errorStrategy 'ignore'

    if (params.keep_intermediate) {
        publishDir "$params.output_directory", mode: 'copy', pattern: "*"
    }

    input:
    tuple val(name), val(read1), val(read2)

    output:
    tuple val(name), path("${name}.kmc_*"), path("${name}.hist.txt"), optional: true

    script:
    """
    echo ${read1} > ${name}.read_manifest.txt
    echo ${read2} >> ${name}.read_manifest.txt
    kmc -hp -k${params.kmer_size} -t${task.cpus} -ci2 -cs65535 -m16 @${name}.read_manifest.txt ${name} .
    kmc_tools -hp transform ${name} histogram ${name}.hist.txt
    """
}

process filter_kmc {
    if (params.keep_intermediate) {
        publishDir "$params.output_directory", mode: 'copy', pattern: "*"
    }

    input:
    tuple val(name), path(db_paths), val(low), val(high)

    output:
    tuple val(name), path("${name}.kmers.fa"), optional: true

    script:
    """
    kmc_tools -hp transform ${name} -ci$low -cx$high dump ${name}.kmers.txt
    awk '{print ">"NR"\\n"\$1}' ${name}.kmers.txt > ${name}.kmers.fa
    """
}

if (params.counter == "kmc") {
} else {
    error "Unsupported kmer counter: ${params.counter}"
}

process estimate_coverage {
    errorStrategy 'ignore'

    if (params.keep_intermediate) {
        publishDir "$params.output_directory", mode: 'copy', pattern: "*"
    }

    input:
    tuple val(name), path(db_paths), path(histogram_path)

    output:
    tuple val(name), path(db_paths), path("${name}.coverage.txt"), optional: true

    script:
    """
    estimate_coverage ${histogram_path} > ${name}.coverage.txt
    """
}

process sourmash_sketch {
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    tuple val(name), path(kmers_path)

    output:
    path("${name}.sig"), optional: true

    // script:
    // """
    // sourmash sketch dna -p 'k=$params.kmer_size,scaled=1000' ${kmers_path} -o${name}.sig
    // """

    script:
    """
    sourmash scripts singlesketch -p 'k=$params.kmer_size,scaled=1000,dna' -o${name}.sig ${kmers_path}
    """
}

process sourmash_merge {
    publishDir "$params.output_directory", mode: 'copy', pattern: "merged.sig"

    input:
    path sigs  // gather all emitted sketches

    output:
    path "merged.sig"

    script:
    """
    sourmash signature cat ${sigs.join(' ')} -o merged.sig
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

    lines = Channel
        .fromPath(params.manifest)
        .splitText()
        .filter { it.trim() }

    counts_input = lines.map { line ->
        def p = line.trim().split(',')
        if (p.size() != 3)
            error "Invalid line in CSV: '${line}'"
        def name = p[0]
        def read1 = p[1]
        def read2 = p[2]
        tuple(name, read1, read2)
    }

    if (params.counter == "kmc") {
        counts = count_kmc(counts_input)
    }

    countsWithCoverage = estimate_coverage(counts)

    filters = countsWithCoverage.map { 
        name, db_paths, coverage_path ->
        def coverage = new File(coverage_path.toString()).text.trim().toFloat()
        def low = (int) Math.ceil(coverage * params.low)
        def high = (int) Math.floor(coverage * params.high)
        [name, db_paths, low, high]
    }

    if (params.counter == "kmc") {
        kmers = filter_kmc(filters)
    }

    sketches = sourmash_sketch(kmers)

    merged_sketch = sourmash_merge(sketches.collect())
}
