set +e
samtools faidx $1
gatk CreateSequenceDictionary -R $1
