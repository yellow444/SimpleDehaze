using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

using Brush = System.Windows.Media.Brush;
using Brushes = System.Windows.Media.Brushes;
using Cursors = System.Windows.Input.Cursors;
using MouseEventArgs = System.Windows.Input.MouseEventArgs;
using Point = System.Windows.Point;

namespace SimpleDeHaze.Gui.Modern
{
    /// <summary>
    /// Общее состояние просмотра для нескольких панелей: масштаб + точка изображения,
    /// стоящая в центре вьюпорта. Один экземпляр на все панели -> синхронные зум и панорама.
    /// В отличие от WinForms-версии храним не пиксельное смещение, а точку в координатах
    /// изображения: панели разной ширины тогда показывают один и тот же участок кадра.
    /// </summary>
    public sealed class ZoomState
    {
        /// <summary>Масштаб (1 = пиксель в пиксель). Действует, когда <see cref="Fitted"/> = false.</summary>
        public double Scale = 1;

        /// <summary>Точка изображения (в его координатах), которую держим в центре вьюпорта.</summary>
        public double CenterX, CenterY;

        /// <summary>true - каждая панель вписывает свой кадр сама (стартовый режим).</summary>
        public bool Fitted = true;

        public event Action? Changed;

        public void Raise() => Changed?.Invoke();

        /// <summary>Вернуться к режиму «вписать» во всех панелях.</summary>
        public void Fit() { Fitted = true; Raise(); }
    }

    /// <summary>
    /// Панель просмотра изображения: колесо - зум к курсору, перетаскивание - панорама,
    /// двойной клик - вписать, клик без движения - <see cref="Clicked"/>.
    /// Состояние берётся из общего <see cref="ZoomState"/>, поэтому остальные панели
    /// с тем же состоянием повторяют масштаб и сдвиг. Аналог WinForms-класса
    /// <see cref="SimpleDeHaze.Gui.ZoomImageView"/> в старом GUI.
    /// </summary>
    public sealed class ZoomView : FrameworkElement
    {
        private const double ZoomStep = 1.15, MinScale = 0.02, MaxScale = 60.0;

        private Point _last, _press;
        private bool _drag, _moved;

        public ZoomView()
        {
            ClipToBounds = true;
            Focusable = false;
        }

        /// <summary>Клик по панели без перетаскивания (используется для загрузки файла).</summary>
        public event EventHandler? Clicked;

        // ---------- свойства ----------

        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(
            nameof(Source), typeof(ImageSource), typeof(ZoomView),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));

        public ImageSource? Source
        {
            get => (ImageSource?)GetValue(SourceProperty);
            set => SetValue(SourceProperty, value);
        }

        public static readonly DependencyProperty StateProperty = DependencyProperty.Register(
            nameof(State), typeof(ZoomState), typeof(ZoomView),
            new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, OnStateChanged));

        /// <summary>Общее состояние. Панели с одним и тем же объектом двигаются синхронно.</summary>
        public ZoomState? State
        {
            get => (ZoomState?)GetValue(StateProperty);
            set => SetValue(StateProperty, value);
        }

        public static readonly DependencyProperty BackgroundProperty = DependencyProperty.Register(
            nameof(Background), typeof(Brush), typeof(ZoomView),
            new FrameworkPropertyMetadata(Brushes.Black, FrameworkPropertyMetadataOptions.AffectsRender));

        public Brush? Background
        {
            get => (Brush?)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        public static readonly DependencyProperty EmptyHintProperty = DependencyProperty.Register(
            nameof(EmptyHint), typeof(string), typeof(ZoomView),
            new FrameworkPropertyMetadata("", FrameworkPropertyMetadataOptions.AffectsRender));

        /// <summary>Текст по центру, когда изображения нет.</summary>
        public string EmptyHint
        {
            get => (string)GetValue(EmptyHintProperty);
            set => SetValue(EmptyHintProperty, value);
        }

        public static readonly DependencyProperty FixedScaleProperty = DependencyProperty.Register(
            nameof(FixedScale), typeof(double), typeof(ZoomView),
            new FrameworkPropertyMetadata(double.NaN, FrameworkPropertyMetadataOptions.AffectsRender));

        /// <summary>
        /// Если задан (не NaN) - панель игнорирует общий масштаб и рисует с этим (лупа 100 % = 1.0),
        /// но продолжает следовать за общей точкой центра. NaN - обычный синхронный режим.
        /// </summary>
        public double FixedScale
        {
            get => (double)GetValue(FixedScaleProperty);
            set => SetValue(FixedScaleProperty, value);
        }

        public static readonly DependencyProperty InteractiveProperty = DependencyProperty.Register(
            nameof(Interactive), typeof(bool), typeof(ZoomView), new PropertyMetadata(true));

        /// <summary>false - панель только показывает (лупа): мышь не меняет общее состояние.</summary>
        public bool Interactive
        {
            get => (bool)GetValue(InteractiveProperty);
            set => SetValue(InteractiveProperty, value);
        }

        private static void OnStateChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var v = (ZoomView)d;
            if (e.OldValue is ZoomState o) o.Changed -= v.InvalidateVisual;
            if (e.NewValue is ZoomState n) n.Changed += v.InvalidateVisual;
        }

        // ---------- геометрия ----------

        /// <summary>Размер изображения в DIP. false - изображения нет либо оно вырождено.</summary>
        private bool TryImageSize(out double iw, out double ih)
        {
            iw = ih = 0;
            if (Source is not { } s) return false;
            iw = s.Width; ih = s.Height;
            return iw > 0 && ih > 0;
        }

        /// <summary>Масштаб «вписать в панель».</summary>
        private double FitScale(double iw, double ih)
            => ActualWidth < 2 || ActualHeight < 2 ? 1 : Math.Min(ActualWidth / iw, ActualHeight / ih);

        /// <summary>Действующие масштаб и центр: из общего состояния либо из режима «вписать».</summary>
        private (double scale, double cx, double cy) Effective(double iw, double ih)
        {
            var st = State;
            if (st == null || st.Fitted) return (FitScale(iw, ih), iw / 2, ih / 2);
            double scale = double.IsNaN(FixedScale) ? st.Scale : FixedScale;
            return (scale, st.CenterX, st.CenterY);
        }

        /// <summary>Левый верхний угол изображения во вьюпорте.</summary>
        private (double ox, double oy) Origin(double scale, double cx, double cy)
            => (ActualWidth / 2 - cx * scale, ActualHeight / 2 - cy * scale);

        /// <summary>Записать в общее состояние текущий вид (нужно при первом действии из режима «вписать»).</summary>
        private void Adopt(double scale, double cx, double cy)
        {
            var st = State!;
            st.Scale = scale; st.CenterX = cx; st.CenterY = cy; st.Fitted = false;
        }

        // ---------- мышь ----------

        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            if (!Interactive || State is not { } st || !TryImageSize(out double iw, out double ih)) return;

            var (scale, cx, cy) = Effective(iw, ih);
            double ns = Math.Clamp(scale * (e.Delta > 0 ? ZoomStep : 1 / ZoomStep), MinScale, MaxScale);
            var p = e.GetPosition(this);
            var (ox, oy) = Origin(scale, cx, cy);

            // точка изображения под курсором остаётся под курсором
            double ix = (p.X - ox) / scale, iy = (p.Y - oy) / scale;
            Adopt(ns, ix + (ActualWidth / 2 - p.X) / ns, iy + (ActualHeight / 2 - p.Y) / ns);
            st.Raise();
            e.Handled = true;
        }

        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            if (!Interactive) return;

            if (e.ClickCount == 2) { State?.Fit(); e.Handled = true; return; }

            _drag = true; _moved = false;
            _last = _press = e.GetPosition(this);
            CaptureMouse();
            e.Handled = true;
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            if (!_drag || State is not { } st || !TryImageSize(out double iw, out double ih)) return;

            var p = e.GetPosition(this);
            if (!_moved && Math.Abs(p.X - _press.X) + Math.Abs(p.Y - _press.Y) > 3)
            {
                _moved = true;
                Cursor = Cursors.SizeAll;
            }
            if (_moved)
            {
                var (scale, cx, cy) = Effective(iw, ih);
                Adopt(scale, cx - (p.X - _last.X) / scale, cy - (p.Y - _last.Y) / scale);
                st.Raise();
            }
            _last = p;
        }

        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            if (!_drag) return;
            _drag = false;
            ReleaseMouseCapture();
            Cursor = Cursors.Arrow;
            if (!_moved) Clicked?.Invoke(this, EventArgs.Empty);
        }

        protected override void OnRenderSizeChanged(SizeChangedInfo info)
        {
            base.OnRenderSizeChanged(info);
            InvalidateVisual();   // в режиме «вписать» масштаб зависит от размера панели
        }

        // ---------- отрисовка ----------

        protected override void OnRender(DrawingContext dc)
        {
            var area = new Rect(0, 0, ActualWidth, ActualHeight);
            dc.DrawRectangle(Background ?? Brushes.Black, null, area);   // заодно даёт hit-test

            if (!TryImageSize(out double iw, out double ih))
            {
                DrawHint(dc);
                return;
            }

            var (scale, cx, cy) = Effective(iw, ih);
            var (ox, oy) = Origin(scale, cx, cy);

            // при увеличении показываем реальные пиксели, при уменьшении - сглаживаем.
            // Свойство AffectsRender, поэтому трогаем его только при реальной смене режима.
            var mode = scale >= 1 ? BitmapScalingMode.NearestNeighbor : BitmapScalingMode.HighQuality;
            if (RenderOptions.GetBitmapScalingMode(this) != mode)
                RenderOptions.SetBitmapScalingMode(this, mode);
            dc.DrawImage(Source, new Rect(ox, oy, iw * scale, ih * scale));
        }

        private void DrawHint(DrawingContext dc)
        {
            if (string.IsNullOrEmpty(EmptyHint)) return;
            var text = new FormattedText(EmptyHint, CultureInfo.CurrentUICulture, System.Windows.FlowDirection.LeftToRight,
                new Typeface("Segoe UI"), 12.5, Brushes.Gray, VisualTreeHelper.GetDpi(this).PixelsPerDip)
            {
                MaxTextWidth = Math.Max(40, ActualWidth - 32),
                TextAlignment = TextAlignment.Center
            };
            dc.DrawText(text, new Point((ActualWidth - text.Width) / 2, (ActualHeight - text.Height) / 2));
        }
    }
}
