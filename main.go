package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"time"

	"github.com/gen2brain/malgo"
)

const (
	segmentSize      = 2048
	hopSize          = 1024
	numChromaBands   = 12
	numChromaVectors = 3
)

func captureAndFingerprint(pCapturedSamples []byte) ([]byte, error) {
	// Step 1: Convert PCM samples to floating-point values (-1.0 to 1.0)
	audioData := convertToFloat32(pCapturedSamples)

	// Step 2: Divide audio signal into segments
	segments := splitIntoSegments(audioData)

	// Step 3: Apply windowing function to each segment
	applyWindowFunction(segments)

	// Step 4: Compute Short-Time Fourier Transform (STFT) for each segment
	stft := computeSTFT(segments)

	// Step 5: Convert frequency domain representation to chroma features
	chromaFeatures := computeChromaFeatures(stft)

	// Step 6: Quantize chroma features to a discrete representation
	quantizedChroma := quantizeChromaFeatures(chromaFeatures)

	// Step 7: Encode discrete chroma features into fingerprint representation
	fingerprint := encodeFingerprint(quantizedChroma)

	return fingerprint, nil
}

func convertToFloat32(samples []byte) []float32 {
	numSamples := len(samples) / 2 // Assuming 16-bit samples

	floatData := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		// Convert two bytes to a signed 16-bit integer
		sample := int16(samples[i*2]) | int16(samples[i*2+1])<<8

		// Normalize to the range of -1.0 to 1.0
		floatData[i] = float32(sample) / float32(math.MaxInt16)
	}

	return floatData
}

func splitIntoSegments(audioData []float32) [][]float32 {
	numSamples := len(audioData)
	segmentCount := (numSamples-segmentSize)/hopSize + 1

	segments := make([][]float32, segmentCount)

	for i := 0; i < segmentCount; i++ {
		start := i * hopSize
		end := start + segmentSize

		// Extract the segment from the audio data
		segment := audioData[start:end]

		// Store the segment in the segments slice
		segments[i] = segment
	}

	return segments
}

func applyWindowFunction(segments [][]float32) {
	// Implement the logic to apply a windowing function to each segment
	// You can use popular windowing functions like Hann or Hamming
	// Modify each segment in-place with the applied window function

	segmentSize := len(segments[0])

	// Create a Hann window function
	window := make([]float32, segmentSize)
	for i := 0; i < segmentSize; i++ {
		window[i] = 0.5 * (1 - float32(math.Cos(2*math.Pi*float64(i)/float64(segmentSize-1))))
	}

	// Apply the window function to each segment
	for i := 0; i < len(segments); i++ {
		segment := segments[i]
		for j := 0; j < segmentSize; j++ {
			segment[j] *= window[j]
		}
	}
}

func computeSTFT(segments [][]float32) [][]complex128 {
	// Implement the logic to compute the Short-Time Fourier Transform (STFT) for each segment
	// You can use a Fast Fourier Transform (FFT) implementation
	// Return a 2D slice where each row represents the frequency domain representation of a segment

	segmentSize := len(segments[0])
	numSegments := len(segments)

	stft := make([][]complex128, numSegments)

	for i := 0; i < numSegments; i++ {
		segment := segments[i]

		// Apply the FFT to the segment
		frequencyDomain := make([]complex128, segmentSize)
		for j := 0; j < segmentSize; j++ {
			frequencyDomain[j] = complex(float64(segment[j]), 0)
		}
		fft(frequencyDomain)

		stft[i] = frequencyDomain
	}

	return stft
}

// FFT implementation (e.g., Cooley-Tukey algorithm)
func fft(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}

	// Split into even and odd indices
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Recursive FFT on even and odd parts
	fft(even)
	fft(odd)

	// Combine the results
	for k := 0; k < n/2; k++ {
		t := complex128(cmplx.Rect(1, -2*math.Pi*float64(k)/float64(n))) * odd[k]
		x[k] = even[k] + t
		x[k+n/2] = even[k] - t
	}
}

func computeChromaFeatures(stft [][]complex128) [][]float64 {
	// Implement the logic to convert the frequency domain representation to chroma features
	// You'll need to apply triangular filters and perform magnitude calculations
	// Return a 2D slice where each row represents the chroma features of a segment

	numSegments := len(stft)

	chromaFeatures := make([][]float64, numSegments)

	for i := 0; i < numSegments; i++ {
		segment := stft[i]

		// Compute the magnitudes of each frequency bin in the segment
		magnitudes := make([]float64, len(segment))
		for j := 0; j < len(segment); j++ {
			magnitudes[j] = cmplx.Abs(segment[j])
		}

		// Compute the chroma features using triangular filters
		chroma := computeChromaFromMagnitudes(magnitudes)

		chromaFeatures[i] = chroma
	}

	return chromaFeatures
}

func computeChromaFromMagnitudes(magnitudes []float64) []float64 {
	numBins := len(magnitudes)
	chromaFeatures := make([]float64, numChromaBands)

	binWidth := float64(numBins) / float64(numChromaBands)

	for i := 0; i < numChromaBands; i++ {
		startBin := int(math.Round(float64(i) * binWidth))
		endBin := int(math.Round(float64(i+1) * binWidth))

		sum := 0.0
		for j := startBin; j < endBin; j++ {
			sum += magnitudes[j]
		}

		chromaFeatures[i] = sum
	}

	return chromaFeatures
}

func quantizeChromaFeatures(chromaFeatures [][]float64) [][]int {
	// Implement the logic to quantize the chroma features to a discrete representation
	// You can choose to use peak picking or thresholding methods
	// Return a 2D slice where each row represents the quantized chroma features of a segment

	numSegments := len(chromaFeatures)

	quantizedChroma := make([][]int, numSegments)

	for i := 0; i < numSegments; i++ {
		segment := chromaFeatures[i]

		// Apply thresholding to quantize chroma features
		quantizedSegment := make([]int, len(segment))
		for j := 0; j < len(segment); j++ {
			if segment[j] > 0.5 { // Adjust the threshold value as needed
				quantizedSegment[j] = 1
			} else {
				quantizedSegment[j] = 0
			}
		}

		quantizedChroma[i] = quantizedSegment
	}

	return quantizedChroma
}

func encodeFingerprint(quantizedChroma [][]int) []byte {
	// Implement the logic to encode the discrete chroma features into a fingerprint representation
	// You'll need to decide on the fingerprint format and encoding scheme
	// Return the encoded fingerprint as a byte slice

	numSegments := len(quantizedChroma)

	fingerprintSize := numSegments * numChromaBands // Assuming each segment has numChromaBands chroma features
	fingerprint := make([]byte, fingerprintSize)

	for i := 0; i < numSegments; i++ {
		segment := quantizedChroma[i]

		for j := 0; j < numChromaBands; j++ {
			bit := segment[j]

			// Set the corresponding bit in the fingerprint byte
			fingerprint[i*numChromaBands+j] |= byte(bit) << uint(7-j)
		}
	}

	return fingerprint
}

func captureAudio(duration time.Duration, debug bool) ([]byte, error) {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		if debug {
			fmt.Printf("LOG <%v>", message)
		}
	})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = 8000
	deviceConfig.Alsa.NoMMap = 1

	var capturedSampleCount uint32
	pCapturedSamples := make([]byte, 0)

	stopCapture := false // Flag to indicate when to stop capturing

	onRecvFrames := func(pSample2, pSample []byte, framecount uint32) {
		if stopCapture {
			return // If the flag is set, stop capturing frames
		}

		sampleCount := int(framecount) * int(deviceConfig.Capture.Channels) * malgo.SampleSizeInBytes(deviceConfig.Capture.Format)

		newCapturedSampleCount := capturedSampleCount + uint32(sampleCount)
		pCapturedSamples = append(pCapturedSamples, pSample...)
		capturedSampleCount = newCapturedSampleCount

		if time.Duration(capturedSampleCount/deviceConfig.SampleRate)*time.Second >= duration {
			stopCapture = true // Set the flag to stop capturing frames
		}
	}

	captureCallbacks := malgo.DeviceCallbacks{
		Data: onRecvFrames,
	}
	device, err := malgo.InitDevice(ctx.Context, deviceConfig, captureCallbacks)
	if err != nil {
		return nil, err
	}

	err = device.Start()
	if err != nil {
		return nil, err
	}

	// Wait for the specified duration
	time.Sleep(duration)

	stopCapture = true // Set the flag to stop capturing frames

	device.Uninit()

	// Wait for any remaining audio processing
	time.Sleep(100 * time.Millisecond)

	// Convert captured audio to byte buffer
	pcmBuffer := make([]byte, len(pCapturedSamples))
	copy(pcmBuffer, pCapturedSamples)

	return pcmBuffer, nil
}

func main() {
	duration := 5 * time.Second // Specify the desired duration for capturing audio
	debug := true               // Set the debug flag if needed

	// Call the captureAudio function
	audioData, err := captureAudio(duration, debug)
	if err != nil {
		fmt.Println("Error capturing audio:", err)
		return
	}

	fingerprint, err := captureAndFingerprint(audioData)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Println("Fingerprint:", fingerprint)
}
