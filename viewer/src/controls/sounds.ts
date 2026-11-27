/**
 * Footstep, jump and landing sounds, synthesised from filtered noise so there are no audio files
 * to ship or license. Browsers only allow audio after a user gesture, so the context starts on
 * the first click, tap or key press.
 */
export class WalkSounds {
  muted = false;
  private ctx: AudioContext | null = null;
  private noise: AudioBuffer | null = null;

  constructor() {
    const unlock = () => {
      this.ctx ??= new AudioContext();
      if (this.ctx.state === 'suspended') void this.ctx.resume();
    };
    document.addEventListener('pointerdown', unlock);
    document.addEventListener('keydown', unlock);
  }

  /** One footfall on a hard floor; a little random so a walk doesn't sound like a metronome. */
  step(): void {
    this.burst(350 + Math.random() * 150, 0.18 + Math.random() * 0.06, 0.07);
  }

  jump(): void {
    this.burst(900, 0.06, 0.12);
  }

  land(): void {
    this.burst(180, 0.35, 0.14);
  }

  private burst(frequencyHz: number, gain: number, seconds: number): void {
    const ctx = this.ctx;
    if (this.muted || !ctx || ctx.state !== 'running') return;
    this.noise ??= this.whiteNoise(ctx);
    const source = new AudioBufferSourceNode(ctx, { buffer: this.noise });
    const filter = new BiquadFilterNode(ctx, { type: 'bandpass', frequency: frequencyHz, Q: 1.2 });
    const envelope = new GainNode(ctx, { gain: 0 });
    const now = ctx.currentTime;
    envelope.gain.linearRampToValueAtTime(gain, now + 0.005);
    envelope.gain.exponentialRampToValueAtTime(0.0001, now + seconds);
    source.connect(filter).connect(envelope).connect(ctx.destination);
    source.start(now, Math.random() * 0.5, seconds);
  }

  private whiteNoise(ctx: AudioContext): AudioBuffer {
    const buffer = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate); // 1 s
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = Math.random() * 2 - 1;
    return buffer;
  }
}
