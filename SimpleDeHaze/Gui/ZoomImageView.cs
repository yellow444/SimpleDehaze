using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace SimpleDeHaze.Gui
{
    /// <summary>Общее состояние просмотра (масштаб + смещение). Один на все виды -> синхронный зум/панорама.</summary>
    public sealed class ViewState
    {
        public float Scale = 1f;
        public float OffsetX, OffsetY;
        public bool Fitted = true;
        public event Action? Changed;
        public void Raise() => Changed?.Invoke();
    }

    /// <summary>
    /// Просмотр изображения с зумом (колесо), панорамой (перетаскивание) и кликом-без-движения
    /// (событие <see cref="Clicked"/> - используется для загрузки картинки кликом по полю).
    /// </summary>
    public sealed class ZoomImageView : Control
    {
        private Bitmap? _img;
        private readonly ViewState _vs;
        private Point _last, _press;
        private bool _drag, _moved;

        /// <summary>Подсказка, когда изображение не загружено.</summary>
        public string EmptyHint = "";

        /// <summary>Клик по полю без перетаскивания (для загрузки изображения).</summary>
        public event Action? Clicked;

        public ZoomImageView(ViewState vs)
        {
            _vs = vs;
            DoubleBuffered = true;
            BackColor = Color.FromArgb(32, 32, 32);
            _vs.Changed += Invalidate;

            MouseWheel += OnWheel;
            MouseDown += (_, e) => { if (e.Button == MouseButtons.Left) { _drag = true; _moved = false; _last = e.Location; _press = e.Location; } };
            MouseUp += (_, e) => { if (e.Button == MouseButtons.Left) { _drag = false; Cursor = Cursors.Default; if (!_moved) Clicked?.Invoke(); } };
            MouseMove += OnMove;
            Resize += (_, _) => { if (_vs.Fitted) Fit(); else Invalidate(); };
        }

        public Bitmap? Image
        {
            get => _img;
            set { _img?.Dispose(); _img = value; Invalidate(); }
        }

        /// <summary>Вписать изображение в окно (сбрасывает общий масштаб/смещение для всех видов).</summary>
        public void Fit()
        {
            if (_img == null || Width < 2 || Height < 2) return;
            _vs.Scale = Math.Min((float)Width / _img.Width, (float)Height / _img.Height);
            _vs.OffsetX = (Width - _img.Width * _vs.Scale) / 2f;
            _vs.OffsetY = (Height - _img.Height * _vs.Scale) / 2f;
            _vs.Fitted = true;
            _vs.Raise();
        }

        private void OnWheel(object? s, MouseEventArgs e)
        {
            if (_img == null) return;
            float old = _vs.Scale;
            float ns = Math.Clamp(old * (e.Delta > 0 ? 1.15f : 1f / 1.15f), 0.02f, 60f);
            _vs.OffsetX = e.X - (e.X - _vs.OffsetX) * (ns / old);
            _vs.OffsetY = e.Y - (e.Y - _vs.OffsetY) * (ns / old);
            _vs.Scale = ns;
            _vs.Fitted = false;
            _vs.Raise();
        }

        private void OnMove(object? s, MouseEventArgs e)
        {
            if (!_drag) return;
            if (Math.Abs(e.X - _press.X) + Math.Abs(e.Y - _press.Y) > 3) { _moved = true; Cursor = Cursors.SizeAll; }
            if (_moved)
            {
                _vs.OffsetX += e.X - _last.X;
                _vs.OffsetY += e.Y - _last.Y;
                _vs.Fitted = false;
                _vs.Raise();
            }
            _last = e.Location;
        }

        protected override void OnPaintBackground(PaintEventArgs e) => e.Graphics.Clear(BackColor);

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            if (_img == null)
            {
                if (!string.IsNullOrEmpty(EmptyHint))
                {
                    var sz = g.MeasureString(EmptyHint, Font);
                    g.DrawString(EmptyHint, Font, Brushes.Gray, (Width - sz.Width) / 2f, (Height - sz.Height) / 2f);
                }
                return;
            }
            g.InterpolationMode = _vs.Scale >= 1f ? InterpolationMode.NearestNeighbor : InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            g.DrawImage(_img, new RectangleF(_vs.OffsetX, _vs.OffsetY, _img.Width * _vs.Scale, _img.Height * _vs.Scale));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) _img?.Dispose();
            base.Dispose(disposing);
        }
    }
}
