namespace SnowRoader.Models;

public class ModelInput
{
    public byte[] Image { get; set; } = default!;
    public UInt32 LabelAsKey { get; set; }
    public string ImagePath { get; set; } = default!;
    public string Label { get; set; } = default!;
}