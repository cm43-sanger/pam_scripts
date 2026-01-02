#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.help = false
params.manifest = null
params.output_directory = null
params.kmer_size = 21
params.force = false
params.keep_intermediate = false

def invalidChars = ['(', ')', '|', '&', '<', '>', ';', ':', '"', "'", '`', '\\', '*', '?']

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

process sketch {
    cpus 1
    publishDir "$params.output_directory", mode: 'copy', pattern: "*"

    input:
    tuple val(name), path(fa)

    output:
    path("${name}.sig"), optional: true

    script:
    """
    echo "test" $fa
    sourmash sketch dna -p 'k=$params.kmer_size,scaled=1000' "${fa}" -o"${name}.sig"
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

    processedInput = lines.map { line ->
        def p = line.trim().split(',')
        if (p.size() != 2)
            error "Invalid line in CSV: '${line}'"
        def name = p[0]
        // if (invalidChars.any { c -> name.contains(c) }) {
        //     error "Name contains invalid characters ${invalidChars.join('')}:\n${name}"
        // }
        def fa = p[1]
        if (invalidChars.any { c -> fa.contains(c) }) {
            error "Filename contains invalid characters ${invalidChars.join('')}:\n${fa}"
        }
        tuple(name, fa)
    }

    sketches = sketch(processedInput)
}
