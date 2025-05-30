import torch
import numpy as np
import os
import sys

# It's good practice to ensure the src directory is in PYTHONPATH
# In a real execution environment, this might be set externally.
# For this script, we can add it if not already set by an external mechanism.
# Assuming this script is run from the root of the repository, and 'src' is a subdir.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..')) # Adjust if script is not in root
# src_path = os.path.join(project_root, 'src')
# if src_path not in sys.path:
#    sys.path.insert(0, src_path)

# The above path manipulation is commented out as PYTHONPATH is expected to be set
# by the execution environment (e.g., `export PYTHONPATH=/app/src:$PYTHONPATH`)

try:
    from chatterbox.models.s3gen.xvector import CAMPPlus
    # If S3_SR or other constants are needed from chatterbox for defaults:
    # from chatterbox.models.s3tokenizer import S3_SR # S3_SR = 16000
except ModuleNotFoundError as e:
    print(f"Failed to import chatterbox modules. Ensure PYTHONPATH is set correctly.")
    print(f"Example: export PYTHONPATH=/app/src:$PYTHONPATH (if /app is your project root)")
    print(f"Error: {e}")
    sys.exit(1)

def prepare_campplus_for_export():
    """
    Prepares the CAMPPlus model and dummy inputs for ONNX export.
    The actual torch.onnx.export call is commented out.
    """
    print('Instantiating CAMPPlus model...')
    # Default args for CAMPPlus: feat_dim=80, embedding_size=192
    # Output level 'segment' is default, which means it does statistics pooling.
    model = CAMPPlus() 
    model.eval() 
    print('CAMPPlus model instantiated successfully and set to eval mode.')

    # --- Prepare Dummy Inputs ---
    print('Preparing dummy inputs...')
    batch_size = 1 
    sample_rate = 16000 # Consistent with xvector.py's extract_feature default
    num_samples = sample_rate * 1 # 1 second of audio

    dummy_batched_wav = torch.randn(batch_size, num_samples)
    
    print(f'  Dummy batched waveform shape: {dummy_batched_wav.shape}')

    print('Defining CAMPPlusExportWrapper...')
    class CAMPPlusExportWrapper(torch.nn.Module):
        def __init__(self, campplus_model):
            super().__init__()
            self.model = campplus_model
        
        def forward(self, batched_wav_input):
            # CAMPPlus.inference() expects a list of 1D tensors (waveforms).
            # Convert batched_wav (B, L) to a list of (L,) tensors.
            # Note: torch.split returns views, squeeze removes the batch dim for each.
            audio_list_for_inference = [wav.squeeze(0) for wav in torch.split(batched_wav_input, 1, dim=0)]
            
            # The model.inference method internally calls extract_feature (now using librosa)
            # and then self.forward (on the extracted features).
            # It returns the speaker embedding.
            embedding = self.model.inference(audio_list_for_inference) 
            return embedding

    wrapped_model = CAMPPlusExportWrapper(model)
    # The input to the wrapped_model.forward is the dummy_batched_wav directly
    traced_input = (dummy_batched_wav,) 

    # --- Define Export Parameters ---
    onnx_filepath = 'campplus_speaker_encoder.onnx' 
    input_names = ['waveform'] # Input name for the batched waveform
    output_names = ['speaker_embedding']
    
    # Dynamic axes for batch size and number of samples
    dynamic_axes = {
        'waveform': {0: 'batch_size', 1: 'num_samples'}, 
        'speaker_embedding': {0: 'batch_size'} 
    }
    opset_version = 14

    print(f"\n--- ONNX Export Parameters ---")
    print(f"  ONNX file path: {onnx_filepath}")
    print(f"  Input names: {input_names}")
    print(f"  Output names: {output_names}")
    print(f"  Dynamic axes: {dynamic_axes}")
    print(f"  Opset version: {opset_version}")
    print(f"  Model to export: CAMPPlusExportWrapper (wrapping CAMPPlus.inference)")
    print(f"  Traced input shape: {traced_input[0].shape}")

    # --- Commented-Out Export Call ---
    # print(f'\nAttempting ONNX export to {onnx_filepath} (call is commented out)...')
    # try:
    #     torch.onnx.export(
    #         wrapped_model,
    #         traced_input,
    #         onnx_filepath,
    #         input_names=input_names,
    #         output_names=output_names,
    #         dynamic_axes=dynamic_axes,
    #         opset_version=opset_version,
    #         verbose=True # Set to True for detailed ONNX conversion logs if uncommented
    #     )
    #     print(f"ONNX export call (commented out) would have targeted: {os.getcwd()}/{onnx_filepath}")
    # except Exception as e:
    #     print(f"Error during ONNX export (if it were run): {e}")
    #     import traceback
    #     traceback.print_exc()
    
    print(f"\nScript finished. To perform the actual export:")
    print(f"1. Ensure this script ('{os.path.basename(__file__)}') is executable or run with 'python {os.path.basename(__file__)}'.")
    print(f"2. Make sure PYTHONPATH includes the 'src' directory of the chatterbox project.")
    print(f"   Example: export PYTHONPATH=/path/to/your/chatterbox_project/src:$PYTHONPATH")
    print(f"3. Uncomment the 'torch.onnx.export(...)' section in this script.")
    print(f"4. Run the script.")

if __name__ == '__main__':
    prepare_campplus_for_export()
