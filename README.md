### Audio ID Proof of Concept

Samples audio from a PCM default device and returns a fingprint. This is a proof of concept for a song or other audio content identification system.

Use this technique to sample entire songs and store the fingerprints in a database. Then, when you want to identify a song, sample it and compare the fingerprint to the database. I would imagine a vector database would be better suited for this.