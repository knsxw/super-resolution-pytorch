from processor import ReliableSuperResolution

if __name__ == "__main__":
    processor = ReliableSuperResolution()
    input_path = "muaythai.png"  # Update to your image path
    results, psnr_values = processor.process_image_safely(input_path)
    
    print("\nProcessing completed!")
    for name, psnr in psnr_values.items():
        print(f"{name:20s}: PSNR={psnr:.2f}dB")
