using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;

using Microsoft.Win32;

// WinForms и WPF в одном проекте: одноимённые типы разводим явными алиасами на WPF.
using Button = System.Windows.Controls.Button;
using DataFormats = System.Windows.DataFormats;
using DragEventArgs = System.Windows.DragEventArgs;
using KeyEventArgs = System.Windows.Input.KeyEventArgs;
using MessageBox = System.Windows.MessageBox;
using OpenFileDialog = Microsoft.Win32.OpenFileDialog;
using SaveFileDialog = Microsoft.Win32.SaveFileDialog;

namespace SimpleDeHaze.Gui.Modern
{
    public sealed class InverseBoolToVisibility : IValueConverter
    {
        public object Convert(object value, Type t, object p, CultureInfo c)
            => value is true ? Visibility.Collapsed : Visibility.Visible;

        public object ConvertBack(object value, Type t, object p, CultureInfo c)
            => value is Visibility.Collapsed;
    }

    public partial class NewMainWindow : Window
    {
        private readonly MainViewModel _vm = new();

        /// <summary>Одно состояние просмотра на все панели -> синхронные зум и панорама.</summary>
        private readonly ZoomState _view = new();

        public NewMainWindow(string? initialFile = null)
        {
            InitializeComponent();
            DataContext = _vm;
            InputView.State = ResultView.State = GtView.State = LoupeView.State = _view;
            if (initialFile != null && File.Exists(initialFile)) _vm.LoadHazy(initialFile);
            Closed += (_, _) => _vm.Dispose();
            AllowDrop = true;
            Drop += OnDrop;
            PreviewKeyDown += OnKey;
        }

        private void OnKey(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Z) { _vm.LoupeOn = !_vm.LoupeOn; e.Handled = true; }
            else if (e.Key == Key.F9) { _vm.MetricsPanelOpen = !_vm.MetricsPanelOpen; e.Handled = true; }
            else if (e.Key is Key.D0 or Key.NumPad0) { _view.Fit(); e.Handled = true; }   // вписать все панели
        }

        private void OnDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0)
                _vm.LoadHazy(files[0]);
        }

        private void Open_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "Изображения|*.jpg;*.jpeg;*.png;*.bmp;*.tif|Все файлы|*.*" };
            if (dlg.ShowDialog(this) == true) _vm.LoadHazy(dlg.FileName);
        }

        private void OpenGt_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog { Filter = "Изображения|*.jpg;*.jpeg;*.png;*.bmp;*.tif|Все файлы|*.*" };
            if (dlg.ShowDialog(this) == true) _vm.LoadGt(dlg.FileName);
        }

        /// <summary>Клик по панели входа без перетаскивания - выбрать файл (как в старом GUI).</summary>
        private void Input_Click(object sender, EventArgs e)
        {
            if (_vm.InputImage == null) Open_Click(sender, new RoutedEventArgs());
        }

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new SaveFileDialog { Filter = "PNG|*.png|JPEG|*.jpg|BMP|*.bmp", FileName = "dehazed.png" };
            if (dlg.ShowDialog(this) != true) return;
            if (!_vm.SaveResult(dlg.FileName))
                MessageBox.Show(this, "Сначала нажмите «Обработать».", "Нечего сохранять");
        }

        /// <summary>Кнопка «▾» — открыть меню режимов подбора у самой кнопки.</summary>
        private void AutoMenu_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not Button b || b.ContextMenu is null) return;
            b.ContextMenu.PlacementTarget = b;
            b.ContextMenu.Placement = System.Windows.Controls.Primitives.PlacementMode.Bottom;
            b.ContextMenu.DataContext = DataContext;
            b.ContextMenu.IsOpen = true;
        }

        private void Bench_DoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (sender is DataGrid g && g.SelectedItem is BenchRow row)
            {
                var item = _vm.AllMethods.FirstOrDefault(x => x.Method.Name == row.Method);
                if (item != null) _vm.Selected = item;
            }
        }

        private void Csv_Click(object sender, RoutedEventArgs e)
        {
            if (_vm.Bench.Count == 0) { MessageBox.Show(this, "Сначала «Прогнать все методы».", "Пусто"); return; }
            var dlg = new SaveFileDialog { Filter = "CSV|*.csv", FileName = "dehaze_benchmark.csv" };
            if (dlg.ShowDialog(this) != true) return;

            var ci = CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Метод;Оценка;PSNR;PSNR_совмещ;SSIM;CIEDE2000;Дымка_%;Контраст;Цвет_x;natur_dev_own;мс;мс_на_Мп;Ошибка");
            foreach (var r in _vm.Bench)
                sb.AppendLine(string.Join(";", new[]
                {
                    r.Method, r.Score?.ToString("0.##", ci) ?? "", r.Psnr?.ToString("0.##", ci) ?? "",
                    r.PsnrAligned?.ToString("0.##", ci) ?? "", r.Ssim?.ToString("0.###", ci) ?? "",
                    r.Ciede?.ToString("0.##", ci) ?? "", r.HazePct.ToString("0", ci),
                    r.Contrast.ToString("0.##", ci), r.ColorX.ToString("0.##", ci), r.NaturalnessDev.ToString("0.#", ci),
                    r.Ms.ToString(ci), r.MsPerMp.ToString("0", ci), r.Error ?? ""
                }));
            File.WriteAllText(dlg.FileName, sb.ToString(), new System.Text.UTF8Encoding(true));
        }
    }
}
